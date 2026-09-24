"""Inventory every remaining physics identity without treating lineage as approval."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.audit_physics_legacy_reconciliation import audit as reconcile
from scripts.audit_physics_projection_lineage import grouped, signature

ROOT = Path(__file__).resolve().parents[1]


def audit():
    pairs = reconcile(ROOT)
    reports = []
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
                        row_factory=dict_row, options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for pair in pairs['graphs']:
            source_versions = [pair['legacy_version_id']]
            if pair.get('projected_version_id'):
                source_versions.append(pair['projected_version_id'])
            representations = []
            signatures = []
            dependencies = []
            for version in source_versions:
                row = db.execute('SELECT a.artifact_id,a.fqdn,a.status,a.is_publishable,v.version_id,v.content_hash,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (version,)).fetchone()
                if row['status'] != 'draft' or row['is_publishable'] or not row['is_latest']:
                    raise ValueError('Draft inventory changed')
                nodes = db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (version,)).fetchall()
                edges = db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id', (version,)).fetchall()
                try:
                    groups = grouped(nodes)
                    signatures.append(signature(groups))
                    owner = {node:key for key,g in groups.items() for node in g['nodes']}
                    if len(owner) != len(nodes):
                        raise ValueError('Duplicate source node identity')
                    if any(e['source_id'] not in owner or e['target_id'] not in owner for e in edges):
                        raise ValueError('Unknown edge endpoint')
                    actual = {(owner[e['source_id']],owner[e['target_id']]) for e in edges}
                    expected = {(a,b) for a,ga in groups.items() for b,gb in groups.items() if ga['outputs'] & gb['inputs']}
                    topology = dict(complete_expression_dependencies=actual==expected,
                        issue=None if actual==expected else 'Source-step dependency edges differ',
                        missing_edges=sorted(expected-actual), extraneous_edges=sorted(actual-expected))
                except (KeyError, ValueError) as error:
                    if len(signatures) < len(representations) + 1:
                        signatures.append(None)
                    topology = dict(complete_expression_dependencies=None, issue=str(error), missing_edges=None, extraneous_edges=None)
                representations.append(dict(**row, nodes=len(nodes), edges=len(edges), topology=topology))
                served = db.execute("""SELECT DISTINCT a.artifact_id,a.fqdn,v.version_id,v.content_hash,v.trust_tier,d.optional
                    FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id AND v.is_latest
                    JOIN artifacts a ON a.artifact_id=v.artifact_id
                    WHERE d.dependency_artifact_fqdn=%s AND d.dependency_content_hash=%s AND d.dependency_role='cdg'
                    AND a.artifact_kind='cdg' AND a.status='approved' AND a.is_publishable
                    AND EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id)
                    ORDER BY a.fqdn""", (row['fqdn'],row['content_hash'])).fetchall()
                for parent in served:
                    evidence = db.execute("SELECT runner_version,passed,details->'limitations' AS limitations FROM artifact_audit_evidence WHERE version_id=%s AND audit_type='semantic_audit' ORDER BY runner_version", (parent['version_id'],)).fetchall()
                    dependencies.append(dict(source_version_id=version, parent=parent, stored_semantic_evidence=evidence))
            reports.append(dict(derivation_ids=pair['derivation_ids'], representations=representations,
                source_comparison=pair['classification'], source_differences=pair.get('source_differences'),
                exact_grouped_source_scope=len(signatures)==2 and signatures[0] is not None and signatures[0]==signatures[1],
                served_exact_source_dependencies=dependencies,
                eligible_for_approval=False,
                next_action='Freshly review full corrected scope, provider bindings and applicability before staging original-identity revisions.'))
        current = db.execute("SELECT count(*) AS n FROM artifacts WHERE artifact_kind='cdg' AND fqdn LIKE 'physics.%' AND status='draft'").fetchone()['n']
    accounted = [str(r['artifact_id']) for g in reports for r in g['representations']]
    if len(accounted) != len(set(accounted)) or len(accounted) != current:
        raise ValueError('Every remaining draft identity must be accounted exactly once')
    return dict(read_only=True,catalog_mutations=0,approved=False,draft_artifacts=current,source_families=len(reports),
        exact_grouped_source_pairs=sum(g['exact_grouped_source_scope'] for g in reports),
        families_with_served_exact_source_dependency=sum(bool(g['served_exact_source_dependencies']) for g in reports),
        topology_counts=dict(Counter(str(r['topology']['complete_expression_dependencies']) for g in reports for r in g['representations'])),
        families=reports,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['scripts/audit_physics_legacy_reconciliation.py','scripts/audit_physics_projection_lineage.py']},
        limitations=['Stored scope and provenance inventory only; no numerical execution or fresh source-byte qualification.',
                    'Approved derived graphs may implement a restricted specialization. A source dependency alone does not prove full scope or cross-domain validity.',
                    'Historical source defects remain explicit; corrected versions require version-bound qualification under each original identity.'])


if __name__ == '__main__':
    report=audit()
    (ROOT/'docs/reviews/physics_remaining_reuse_scope.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['families','limitations']},default=str))
