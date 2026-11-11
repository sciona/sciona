"""Plan a complete corrected revision under the immutable first-place identity."""
import hashlib
import json
from pathlib import Path
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.competition_import import _canonical
from sciona.guarded_prediction_graphs import EMIT,validate_descriptor
from sciona.nasa_first_complete_graph import build_nasa_first_complete_graph,root_contracts
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.plan_nasa_corrected_intake import check_atoms
from scripts.review_nasa_first_lifecycle_publication import ROOT,sha,require,plan as lifecycle_plan

ARTIFACT_ID='eb1f8b8c-c1cd-59fc-bb55-69b3044af9e7'
FQDN='cdg.competition.solution.kaggle.nasa_pushback_phase1_1st'
SOURCE_VERSION='0e37a912-48ef-5551-90a3-08290faccb9a'
SOURCE_HASH='8d6045e9fed77a170a2b1f4eca2210522cf2eccc5e0b404cfddd8a73ff829e04'


def check_source_anchor(db):
    row=db.execute('SELECT a.artifact_id,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
    require(row and str(row['artifact_id'])==ARTIFACT_ID and row['fqdn']==FQDN and row['content_hash']==SOURCE_HASH,'Original intake identity changed')
    audits=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(SOURCE_VERSION,)).fetchall()
    require(len(audits)==1 and hashlib.sha256(_canonical(audits[0]['details']['snapshot']).encode()).hexdigest()==SOURCE_HASH,'Immutable original snapshot changed')
    failed=db.execute("SELECT passed,status FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-callable-preflight.v1'",(SOURCE_VERSION,)).fetchall()
    require(failed==[dict(passed=False,status='completed')],'Historical failed preflight evidence changed')


def origin_rows(db):
    result={}
    for table in ['artifact_versions','artifact_io_specs','artifact_cdg_nodes','artifact_cdg_edges',
                  'artifact_cdg_bindings','artifact_dependencies','artifact_audit_evidence']:
        key='dependent_version_id' if table=='artifact_dependencies' else 'version_id'
        rows=db.execute(f"SELECT to_jsonb(t)-'created_at'-'updated_at'-'is_latest' AS row FROM {table} t WHERE {key}=%s ORDER BY (to_jsonb(t)-'created_at'-'updated_at'-'is_latest')::text",(SOURCE_VERSION,)).fetchall()
        result[table]=hashlib.sha256(_canonical([r['row'] for r in rows]).encode()).hexdigest()
    return result


def plan():
    directory=ROOT/'docs/reviews';qualification=directory/'competition_nasa_first_complete_graph.json'
    execution=json.loads(qualification.read_text())
    require(execution['passed'] and execution['executed_outer_nodes']==343 and execution['exact_replayed_fit_operands']==21
        and execution['new_native_fits']==0 and execution['exact_native_prediction_comparisons']==64
        and execution['portable_state_produced'] and execution['qualified_section_projection_equal'],'Complete lifecycle evidence differs')
    for name,digest in execution['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Complete graph implementation changed')
    roots={'sciona-matcher':ROOT,'sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/src','sciona-atoms':ROOT.parent/'sciona-atoms/src'}
    for name,digest in execution['source_closure_sha256'].items():
        repository,relative=name.split('/',1)
        require(sha(roots[repository]/relative)==digest,'Lifecycle runtime closure changed')
    require(execution['native_qualification_sha256']==sha(directory/'competition_nasa_first_native_training_graph.json'),'Native qualification changed')
    graph=build_nasa_first_complete_graph();digest=encode_execution_graph(graph)[0]
    require(digest==execution['graph_sha256'],'Complete graph identity changed')
    available=json.loads((directory/'available_feature_provider_plan.json').read_text())['atoms']
    providers=lifecycle_plan()['atoms']+available
    by_runtime={a['runtime_fqdn']:a for a in providers}
    required={node.matched_primitive for node in graph.nodes}
    nested={}
    for key in ['primary_branch','baseline_branch']:
        branch,_=validate_descriptor(graph.metadata[key])
        nested[key]=[dict(by_runtime[node.matched_primitive],node_id=node.node_id) for node in branch.nodes]
        required.update(node.matched_primitive for node in branch.nodes)
    required.add(EMIT)
    atoms=[by_runtime[name] for name in sorted(required)]
    require(len(atoms)==47,'Exact lifecycle provider closure required')
    bindings=[dict(by_runtime[node.matched_primitive],node_id=node.node_id,
        inputs=[p.model_dump(mode='json') for p in node.inputs],outputs=[p.model_dump(mode='json') for p in node.outputs]) for node in graph.nodes]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_source_anchor(db);check_atoms(db,atoms);original=origin_rows(db)
    filename=directory/'competition_nasa_first_corrected_intake_plan.json'
    if filename.exists():require(json.loads(filename.read_text())['original_rows_sha256']==original,'Original historical records changed')
    return dict(artifact_id=ARTIFACT_ID,fqdn=FQDN,version_id=str(uuid5(UUID(ARTIFACT_ID),'corrected-complete-execution:'+digest)),
        graph_sha256=digest,trust_tier=3,original_version_id=SOURCE_VERSION,original_content_hash=SOURCE_HASH,
        original_rows_sha256=original,inputs=root_contracts(graph),
        outputs=[dict(next(p for n in graph.nodes if n.node_id==node for p in n.outputs if p.name==port).model_dump(mode='json'),name=name)
            for node,port,name in [('training/bind_state','state','state'),('inference/format','predictions','predictions')]],
        output_bindings={'state':{'node_id':'training/bind_state','port_name':'state'},'predictions':{'node_id':'inference/format','port_name':'predictions'}},
        atoms=atoms,bindings=bindings,nested_bindings=nested,controller_dependency=by_runtime[EMIT],
        scope=graph.metadata['scope'],planner_sha256=sha(Path(__file__)),
        evidence_sha256={name:sha(directory/name) for name in ['competition_nasa_first_complete_graph.json',
            'competition_nasa_first_lifecycle_served.json','available_feature_provider_served.json']},
        limitations=['Complete training then one-population prediction per call; saved-state inference has separate qualified topology.',
                     'Native fits are compositionally qualified; full combined traversal replays checked fit operands against qualified native checkpoints.',
                     'Original intake snapshot and failed preflight remain immutable; this candidate is not yet approved.'])


if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/competition_nasa_first_corrected_intake_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(nodes=len(result['bindings']),reused_providers=len(result['atoms']),new_providers=0,version_id=result['version_id'],catalog_mutations=0)))
