"""Reconcile apparently uncovered legacy sources with preserved approved projections.

Historical source versions remain candidates after their artifact's corrected
execution becomes approved. Source identity does not approve a legacy graph.
"""
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.audit_physics_legacy_reconciliation import source_inventory,digest

ROOT=Path(__file__).resolve().parents[1]


def inventory(db,version):
    rows=db.execute('SELECT type_signature FROM artifact_cdg_nodes WHERE version_id=%s',(version,)).fetchall()
    signatures=[json.loads(r['type_signature']) if isinstance(r['type_signature'],str) else r['type_signature'] or {} for r in rows]
    bindings=db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',(version,)).fetchall()
    return source_inventory(signatures,bindings)


def audit():
    baseline=ROOT/'docs/reviews/physics_served_coverage_refresh.json'
    families=json.loads(baseline.read_text())['families']
    unresolved=[f for f in families if not f['served_realizations']]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        historical=db.execute("SELECT a.artifact_id,a.fqdn,v.version_id,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_kind='cdg' AND a.fqdn LIKE 'physics.pdg.%' AND a.status='approved' AND a.is_publishable AND NOT v.is_latest ORDER BY a.fqdn,v.version_id").fetchall()
        candidates=[]
        for row in historical:
            inv=inventory(db,row['version_id'])
            if inv['projected'] and len(inv['derivations'])==1:candidates.append((row,inv))
        reports=[]
        for family in unresolved:
            version=family['source_version_id'];legacy=inventory(db,version);matches=[]
            current=db.execute('SELECT a.artifact_id,a.status,a.is_publishable,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(version,)).fetchone()
            for row,other in candidates:
                if legacy['derivations']!=other['derivations']:continue
                gates=dict(same_inference_contracts=legacy['relations']==other['relations'],
                    same_bound_versions=legacy['binding_versions']==other['binding_versions'],
                    all_bindings_active=legacy['all_bindings_active'] and other['all_bindings_active'])
                served=db.execute("SELECT v.version_id,v.content_hash,v.trust_tier FROM artifact_versions v WHERE v.artifact_id=%s AND v.is_latest AND EXISTS (SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=v.artifact_id)",(row['artifact_id'],)).fetchall()
                provenance=db.execute("SELECT d.dependency_content_hash,d.optional FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id AND v.is_latest WHERE v.artifact_id=%s AND d.dependency_artifact_fqdn=%s AND d.dependency_content_hash=%s AND d.dependency_role='cdg'",(row['artifact_id'],row['fqdn'],row['content_hash'])).fetchall()
                matches.append(dict(projected_original=row,comparison=gates,current_served_versions=served,
                    mandatory_original_provenance=bool(provenance) and all(not p['optional'] for p in provenance),
                    exact_source_match=all(gates.values())))
            reports.append(dict(legacy_version_id=version,legacy_current_state=current,derivation_ids=legacy['derivations'],
                source_relation_digest=digest(legacy['relations']),source_binding_digest=digest(legacy['binding_versions']),
                preserved_projection_matches=matches,
                actionable_reuse_candidates=sum(m['exact_source_match'] and len(m['current_served_versions'])==1 and m['mandatory_original_provenance'] for m in matches)))
        counts=db.execute("SELECT a.status,count(*) AS artifacts FROM artifacts a WHERE a.artifact_kind='cdg' AND a.fqdn LIKE 'physics.%' GROUP BY a.status ORDER BY a.status").fetchall()
    return dict(read_only=True,approved=False,catalog_mutations=0,apparently_uncovered_families=len(unresolved),
        families_with_approved_projection_lineage=sum(r['actionable_reuse_candidates']>0 for r in reports),
        physics_artifact_status_counts=counts,families=reports,
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['scripts/audit_remaining_physics_lineage.py','scripts/audit_physics_legacy_reconciliation.py']},
        limitations=['Exact preserved-source matching identifies reuse candidates; it does not approve legacy identities or every historical equation.',
                     'Before promotion, verify corrected semantic scope and execute the stored approved graph under the legacy identity.',
                     'Counts here restrict physics artifacts; the older reconciliation served_cdgs count includes all domains.'])


if __name__=='__main__':
    report=audit()
    (ROOT/'docs/reviews/physics_remaining_preserved_lineage.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:report[k] for k in ['apparently_uncovered_families','families_with_approved_projection_lineage','physics_artifact_status_counts','catalog_mutations']}))
