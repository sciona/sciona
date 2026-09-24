"""Compare legacy split-step CDGs with historical source-step projections.

This is a read-only structural scope audit, not proof validity or approval.
"""
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

ROOT=Path(__file__).resolve().parents[1]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def grouped(nodes):
    groups={}
    for row in nodes:
        s=json.loads(row['type_signature'])
        bindings=s.get('variable_bindings',{})
        key=(bindings.get('derivation_id'),bindings.get('step_id'))
        if not all(key):raise ValueError('Missing source step identity')
        item=groups.setdefault(key,dict(rule=s['inference_rule_id'],bindings=bindings,inputs=set(),outputs=set(),nodes=[]))
        if item['rule']!=s['inference_rule_id'] or item['bindings']!=bindings:raise ValueError('Inconsistent split-step rule or feeds')
        item['inputs'].update(s['inputs']);item['outputs'].update(s.get('outputs',[s['output']]))
        item['nodes'].append(row['node_id'])
    return groups


def signature(groups):
    return {str(key):dict(rule=value['rule'],bindings=value['bindings'],inputs=sorted(value['inputs']),outputs=sorted(value['outputs'])) for key,value in groups.items()}


def audit():
    readiness=json.loads((ROOT/'docs/reviews/physics_graph_readiness_v2.json').read_text())
    drafts=[g for g in readiness['graphs'] if not g['source_step_projection']]
    reports=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        # Historical projected versions remain the comparison basis after their
        # artifact's latest version becomes an executable numerical revision.
        rows=db.execute("SELECT v.version_id::text,v.artifact_id::text,v.content_hash,a.fqdn FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE a.artifact_kind='cdg' AND a.fqdn LIKE 'physics.pdg.%' AND EXISTS (SELECT 1 FROM artifact_cdg_nodes n WHERE n.version_id=v.version_id AND n.type_signature LIKE '%source_pdg_step_id%') ORDER BY v.version_id").fetchall()
        projected=[]
        for version in rows:
            nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(version['version_id'],)).fetchall()
            try:groups=grouped(nodes)
            except (KeyError,ValueError):continue
            # Normalized expression substitutions are a distinct review layer.
            if any('source_expression_substitutions' in json.loads(n['type_signature']) for n in nodes):continue
            projected.append((version,groups))
        for draft in drafts:
            live=db.execute('SELECT a.status,a.is_publishable,v.content_hash,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s AND a.artifact_id=%s',(draft['version_id'],draft['artifact_id'])).fetchone()
            if live!=dict(status='draft',is_publishable=False,content_hash=draft['content_hash'],is_latest=True):raise ValueError('Readiness state changed')
            nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(draft['version_id'],)).fetchall()
            try:groups=grouped(nodes)
            except (KeyError,ValueError) as error:
                reports.append(dict(artifact_id=draft['artifact_id'],version_id=draft['version_id'],status='unresolved_source_identity',reason=str(error)));continue
            matches=[]
            for parent,other in projected:
                if parent['artifact_id']==draft['artifact_id'] or signature(groups)!=signature(other):continue
                state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(parent['artifact_id'],)).fetchone()
                latest=db.execute('SELECT version_id::text,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',(parent['artifact_id'],)).fetchall()
                served=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s',(parent['artifact_id'],)).fetchone()['n']
                matches.append(dict(**parent,current_state=state,latest_versions=latest,served=served==1,
                    source_steps=len(other),source_nodes=sum(len(v['nodes']) for v in other.values())))
            reports.append(dict(artifact_id=draft['artifact_id'],version_id=draft['version_id'],content_hash=draft['content_hash'],
                status='exact_grouped_source_scope' if matches else 'no_exact_grouped_source_scope',
                legacy_nodes=len(nodes),source_steps=len(groups),grouped_scope_sha256=digest(signature(groups)),matches=matches,
                complete_stored_expression_gates=bool(draft['bindings']) and draft['eligible_expression_bindings']==draft['bindings']))
    return dict(read_only=True,catalog_mutations=0,approved=False,legacy_draft_graphs=len(drafts),
        exact_grouped_scope_graphs=sum(bool(r.get('matches')) for r in reports),
        graphs_with_served_projected_family=sum(any(m['served'] for m in r.get('matches',[])) for r in reports),graphs=reports,
        limitations=['Grouping verifies the same derivation IDs, source step IDs, inference rules, feeds and complete input/output expression sets. It does not assert the legacy split-node graph is executable or semantically valid.',
            'Historical source projections may preserve defective mathematics. Their corrected executable revisions require explicit correction and interpretation lineage, exact provider bindings and independent execution evidence.',
            'Source byte freshness, graph-edge scope and transaction/serving gates remain separate requirements before any legacy-identity promotion.'],
        readiness_sha256=hashlib.sha256((ROOT/'docs/reviews/physics_graph_readiness_v2.json').read_bytes()).hexdigest(),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    report=audit()
    (ROOT/'docs/reviews/physics_projection_lineage.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['graphs','limitations']}))
