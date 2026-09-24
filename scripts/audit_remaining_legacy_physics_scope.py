"""Qualify legacy pairwise scope against approved projected physics families."""
import argparse
from collections import Counter
import importlib
import itertools
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.audit_physics_projection_lineage import ROOT,grouped,signature,digest

FAMILIES={'820976':'two_body','000015':'momentum','681943':'parallel_resistance','918264':'projectile','332170':'constant_acceleration'}


def require(condition,message):
    if not condition:raise ValueError(message)


def scope(nodes,edges):
    groups=grouped(nodes);owner={node:key for key,g in groups.items() for node in g['nodes']}
    require(len(owner)==len(nodes),'Duplicate node identity')
    actual=set()
    for edge in edges:
        require(edge['source_id'] in owner and edge['target_id'] in owner,'Unknown edge endpoint')
        actual.add((owner[edge['source_id']],owner[edge['target_id']]))
    expected={(a,b) for a,ga in groups.items() for b,gb in groups.items() if ga['outputs'] & gb['inputs']}
    require(not actual-expected,'Unexpected source-step dependency edges')
    pairs={key:[] for key in groups}
    for node in nodes:
        s=json.loads(node['type_signature']);bindings=s['variable_bindings'];key=(bindings['derivation_id'],bindings['step_id'])
        pairs[key].extend(itertools.product(s['inputs'],s.get('outputs',[s['output']])))
    return groups,pairs,sorted(actual),sorted(expected-actual)


def audit(source):
    lineage=json.loads((ROOT/'docs/reviews/physics_remaining_preserved_lineage.json').read_text())
    selected=[]
    for item in lineage['families']:
        matches=[m for m in item['preserved_projection_matches'] if m['exact_source_match'] and m['mandatory_original_provenance'] and len(m['current_served_versions'])==1]
        require(len(matches)==1 and len(item['derivation_ids'])==1,'Unique preserved projection required')
        match=matches[0];parent=match['projected_original']
        legacy=dict(artifact_id=item['legacy_current_state']['artifact_id'],version_id=item['legacy_version_id'])
        selected.append((legacy,parent,FAMILIES[item['derivation_ids'][0]]))
    require(len(selected)==5,'Expected five remaining legacy scopes')
    reports=[]
    for legacy,parent,family in selected:
        reviewer=importlib.import_module('scripts.review_'+family+'_revision')
        publisher=importlib.import_module('scripts.promote_'+family+'_revision')
        qualification=reviewer.review(source)
        with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
            graph,imported=publisher.check_publication(db,qualification)
            require(imported['artifact_id']==parent['artifact_id'],'Approved family identity differs')
            state=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s AND a.artifact_id=%s',(legacy['version_id'],legacy['artifact_id'])).fetchone()
            require(state and state['status']=='draft' and not state['is_publishable'] and state['is_latest'],'Legacy source state changed')
            legacy['content_hash']=state['content_hash']
            values=[]
            for version in [legacy['version_id'],parent['version_id']]:
                nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(version,)).fetchall()
                edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id',(version,)).fetchall()
                groups,pairs,relations,missing=scope(nodes,edges)
                values.append((groups,pairs,relations,missing))
            old,new=values
            require(signature(old[0])==signature(new[0]),'Grouped source scope changed')
            require(old[2]==new[2],'Grouped edge lineage differs')
            expected_missing=[(('332170','1044455'),('332170','2562830')),
                              (('332170','7173255'),('332170','2812689'))] if family=='constant_acceleration' else []
            require(old[3]==new[3]==expected_missing,'Unreviewed inherited topology gap')
            for key in old[1]:
                require(Counter(old[1][key])==Counter(new[1][key]),'Split input-output pair multiplicity differs')
            reports.append(dict(family=family,legacy_artifact_id=legacy['artifact_id'],legacy_version_id=legacy['version_id'],legacy_content_hash=legacy['content_hash'],
                projected_artifact_id=parent['artifact_id'],projected_source_version_id=parent['version_id'],projected_source_content_hash=parent['content_hash'],
                approved_execution_version_id=imported['version_id'],approved_execution_graph_sha256=imported['graph_sha256'],
                complete_pairwise_scope=True,source_dependency_edges_verified=not old[3],stored_dependency_edges_match=True,
                inherited_missing_expression_dependency_edges=json.loads(json.dumps(old[3])),
                topology_disposition='Preserve historical omissions; reuse the independently qualified corrected complete family execution.' if old[3] else 'Complete implied source dependency edges verified.',
                source_steps=len(old[0]),grouped_edges=len(old[2]),
                legacy_nodes=sum(len(group['nodes']) for group in old[0].values()),approved_provider_count=len(graph.nodes),new_atoms_required=0,
                qualification=qualification,qualification_sha256=digest(qualification),
                next_action='Stage an explicitly provenance-bound executable revision under the legacy artifact identity, preserving its complete historical rows and parent correction/interpretation limits.'))
    return dict(passed=True,approved=False,catalog_mutations=0,qualified_legacy_scopes=len(reports),graphs=reports,
        limitations=['Exact source-step rule/feed/expression pairing and inter-step dependency coverage only; legacy split-node semantics are not asserted valid.',
            'Approved family review is freshly revalidated, including source interpretation/correction and existing numerical or symbolic evidence.',
            'Legacy-identity staging, stored execution, corruption/rollback/publication/served verification remain required.'],
        validator_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/remaining_legacy_physics_execution_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,qualified_legacy_scopes=len(report['graphs']),catalog_mutations=0)))
