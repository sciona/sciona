"""Read-only source/runtime/provider review; catalog promotion is a later gate."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.competition_import import _canonical
from sciona.ghost.registry import REGISTRY
from sciona.stacking_complete_graph import build_complete_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.promote_otto_execution import check_hashes, provider_ports, boundary_ports
from scripts.review_otto_execution import SOURCE_VERSION,SOURCE_HASH,STAGE_MAPPING
from scripts.review_otto_candidate_runtime import review as runtime_review
from scripts.stage_remaining_legacy_physics_revisions import history

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def review():
    reviews=ROOT/'docs/reviews'
    execution_path=reviews/'competition_otto_decomposed_complete_execution.json'
    execution=json.loads(execution_path.read_text())
    require(execution['status']=='passed' and execution['approved'] is False and execution['catalog_mutations']==0,
            'Unapproved complete execution evidence required')
    graph=build_complete_graph();digest,nodes,edges=encode_execution_graph(graph)
    require(execution['serialized_graph_sha256']==digest,'Complete graph identity differs')
    expected=dict(actual_runner_nodes=62,actual_graph_edges=len(graph.edges),source_entries=33,supplemental_blocks=7,
                  first_level_columns=293,tree_columns=341,neural_columns=354,synthetic_queries_correct=9,
                  native_meta_tuning_fits=8800,native_meta_refits=1100,
                  retained_blend_exact_match=True,decimal_oracle_precision=80,strict_json_result=True)
    require(all(execution[k]==v for k,v in expected.items()),'Complete native qualification differs')
    require(execution['decimal_oracle_rtol']==2e-13 and execution['decimal_oracle_atol']==1e-15,
            'Independent numerical tolerance differs')
    calls=execution['completed_native_meta_bags']
    require(len(calls)==27,'Complete meta execution inventory required')
    for family,runs in (('xgboost',250),('lasagne_neural',600),('adaboost_extratrees',250)):
        selected=[c for c in calls if c['family']==family]
        require(len(selected)==9 and all(c['fitted_models']==runs for c in selected),'Full native family bags required')
        require([(c['fitting_rows'],c['prediction_rows']) for c in selected]==
                [(1012,338),(1012,338),(1013,337),(1013,337)]*2+[(1350,9)],'Native fold/refit populations differ')
    required={'sciona/stacking_complete_graph.py','sciona/stacking_first_graph.py','sciona/stacking_first_operations.py',
              'sciona/stacking_supplemental_graph.py','sciona/stacking_supplemental_operations.py',
              'sciona/stacking_meta_graph.py','sciona/stacking_meta_operations.py','sciona/probability_ensemble.py',
              'scripts/otto_synthetic.py','scripts/validate_stacking_complete_graph.py'}
    require(required<=set(execution['implementation_sha256']),'Incomplete implementation coverage')
    check_hashes(ROOT,execution['implementation_sha256'])
    runtime=runtime_review()
    require(execution['candidate_runtime_evidence_sha256']==runtime['evidence_sha256'],'Native candidate closure changed')
    # Historical source/computation records remain immutable. Their obsolete
    # native paths are historical evidence, not current-runtime qualification.
    retained=json.loads((reviews/'competition_otto_semantic_review.json').read_text())
    require(retained['source_version_id']==SOURCE_VERSION and retained['source_hash']==SOURCE_HASH,
            'Retained source identity differs')
    require(retained['intake_stage_mapping']==STAGE_MAPPING,'Retained six-stage corrections differ')
    check_hashes(reviews,retained['evidence_sha256'])
    check_hashes(ROOT,retained['auxiliary_sha256'])
    retained_execution=json.loads((reviews/'competition_otto_graph_execution.json').read_text())
    native_sources={name for name in retained_execution['code_sha256'] if name.startswith('sciona/otto_')}
    require(native_sources<=set(execution['implementation_sha256']),'Underlying native implementation coverage missing')
    directory=ROOT.parent/'sciona-atoms-ml'
    parsed=_parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',directory),artifact_root=directory/'src/sciona/atoms')
    specs={};interfaces={};providers={};provider_interfaces={}
    for node in graph.nodes:
        matches=[s for s in parsed if s.import_module+'.'+s.source_symbol==node.matched_primitive]
        require(len(matches)==1,'Unique registered provider required')
        spec=matches[0];module=importlib.import_module(spec.import_module);fn=getattr(module,spec.source_symbol)
        require(Path(inspect.getfile(inspect.unwrap(fn))).resolve()==spec.file_path.resolve(),'Runtime provider source differs')
        require(sha(spec.file_path)==execution['provider_sha256'][spec.import_module],'Provider code changed')
        witness=REGISTRY[node.matched_primitive]['witness']
        require(witness is getattr(module,spec.witness_name),'Registered witness differs')
        require(list(inspect.signature(witness).parameters)==[p.name for p in node.inputs],'Witness parameter order differs')
        ports=provider_ports(node,fn)
        serialized={direction:[p.model_dump(mode='json') for p in values] for direction,values in ports.items()}
        if spec.fqdn in provider_interfaces:
            require(provider_interfaces[spec.fqdn]==serialized,'Reused provider port contracts differ')
        provider_interfaces[spec.fqdn]=serialized
        specs[node.node_id]=spec;interfaces[node.node_id]=ports
        providers[spec.fqdn]=dict(version_id=str(spec.version_id),content_hash=spec.content_hash,
            runtime_fqdn=node.matched_primitive,source_sha256=sha(spec.file_path),ports=serialized,
            witness_scope='Stage kinds only; runtime validation mandatory, no static shape-proof claim.')
    require(len(providers)==27,'Expected reusable provider inventory differs')
    symbolic={};remaining={n.node_id:n for n in graph.nodes}
    while remaining:
        ready=[n for n in remaining.values() if all((e.source_id,e.output_name) in symbolic
               for e in graph.edges if e.target_id==n.node_id)]
        require(bool(ready),'Graph witness dataflow is cyclic or incomplete')
        for node in sorted(ready,key=lambda n:n.node_id):
            incoming={e.input_name:(e.source_id,e.output_name) for e in graph.edges if e.target_id==node.node_id}
            values={p.name:symbolic[incoming[p.name]] if p.name in incoming else object() for p in node.inputs}
            witness=REGISTRY[node.matched_primitive]['witness']
            output=witness(**values)
            if len(node.outputs)>1:
                require(type(output) is tuple and len(output)==len(node.outputs),'Witness output arity differs')
                outputs=output
            else:outputs=(output,)
            symbolic.update({(node.node_id,p.name):v for p,v in zip(node.outputs,outputs)})
            del remaining[node.node_id]
    require(symbolic[('meta_result','result')]=={'kind':'Otto.MetaResult'},'Final stage witness differs')
    boundary=boundary_ports(graph,interfaces)
    require([p.name for p in boundary['input']]==['payload'] and [p.name for p in boundary['output']]==['result'],
            'Complete public boundary differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        rows=db.execute("SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.is_latest,v.content_hash,e.details FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id) WHERE v.version_id=%s AND e.runner_version='competition-intake.v1'",(SOURCE_VERSION,)).fetchall()
        require(len(rows)==1,'One preserved source intake required')
        source=rows[0];snapshot=source['details']['snapshot']
        require(source['content_hash']==SOURCE_HASH and hashlib.sha256(_canonical(snapshot).encode()).hexdigest()==SOURCE_HASH,
                'Original source snapshot changed')
        stages=snapshot['template']['stages']
        require(len(stages)==6 and {s['stage_id'] for s in stages}==set(STAGE_MAPPING),'Original six-stage inventory differs')
        corrected=str(uuid5(UUID(source['artifact_id']),'otto-decomposed-original:'+digest))
        latest=db.execute('SELECT version_id::text,content_hash,derives_from::text,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',
                          (source['artifact_id'],)).fetchall()
        require(len(latest)==1,'One latest source identity version required')
        if source['status']=='draft':
            require(not source['is_publishable'] and source['is_latest'] and latest[0]['version_id']==SOURCE_VERSION,
                    'Unpromoted original source selection differs')
        else:
            require(source['status']=='approved' and source['is_publishable'] and not source['is_latest'],
                    'Original identity state differs')
            require(latest[0]==dict(version_id=corrected,content_hash=digest,derives_from=SOURCE_VERSION,trust_tier=3),
                    'Corrected original-identity selection differs')
        preserved=history(db,SOURCE_VERSION)
    return dict(status='complete_execution_scope_reviewed',read_only=True,catalog_mutations=0,eligible_for_approval=False,
        proposed_tier=3,artifact_id=source['artifact_id'],original_version_id=SOURCE_VERSION,original_content_hash=SOURCE_HASH,
        corrected_version_id=corrected,
        original_history_sha256=preserved,graph_sha256=digest,nodes=len(graph.nodes),edges=len(graph.edges),
        source_stage_corrections=STAGE_MAPPING,providers=providers,witness_stage_flow_verified=True,
        reuse_scope=['General probability-bag averaging is separately callable with arbitrary supported class/run counts.',
                    'Source-policy adapters accept caller-owned numerical features, identities, folds and model controls; application feature extraction and class meaning remain external.',
                    'The full graph retains nine classes, nonnegative tabular inputs, five-fold first-level prediction and separate four-fold meta selection.',
                    'This qualified graph shares one KNN bank across entries 24 through 33 and therefore requires identical controls for those entries; alternate group compilations are not qualified by this native run.'],
        limitations=['Automated Tier 3 candidate only; no Tier 1 human review or Tier 2 usage qualification.',
                     'Historical feature-width, implementation and training-volume ambiguities remain; no winning-performance claim.',
                     'Fixed transductive embedding and non-nested upstream producers limit deployment and interpretation of selection losses.',
                     'Uncached provisioned in-process execution only; no static shape proof, persistent model state or clean-install claim.'],
        required_before_promotion=['Stage a corrected original-identity revision and exact provider versions without changing original history or latest selection.',
                                  'Qualify stored graph/provider contracts, corruption rejection and transaction rollback.',
                                  'Publish atomically at Tier 3, verify zero-change repeat publication and fresh served selection/execution.'],
        evidence_sha256={p.name:sha(p) for p in (execution_path,reviews/'competition_otto_semantic_review.json',
                                               reviews/'competition_otto_candidate_runtime_review.json')},
        reviewer_sha256=sha(__file__))


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/competition_otto_decomposed_complete_scope.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','eligible_for_approval','nodes','edges','proposed_tier')}))
