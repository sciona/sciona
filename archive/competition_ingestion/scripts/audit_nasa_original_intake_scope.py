"""Read-only source/intake reconciliation against approved population graph regions."""
import argparse
import ast
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.competition_import import _canonical
from sciona.nasa_lifecycle_graphs import build_nasa_training_graph,build_nasa_inference_graph
from scripts.plan_nasa_population_drafts import ROOT,plan,sha
from scripts.check_nasa_population_catalog import check_staged
from scripts.review_conditional_correction import SOURCE_VERSION,SOURCE_HASH,require

REGIONS={
    'feature_engineering':dict(training=['domain_train','domain_vocabulary','domain_train_departures',
        'domain_train_categories','domain_train_history','domain_train_join','domain_train_arrays','domain_handoff','training'],
        inference=['domain_query','domain_query_departures','domain_query_categories','domain_query_history',
            'domain_query_join','domain_query_arrays','query']),
    'primary_regression':dict(training=['split','internal_fit','internal_predict','internal_round','internal_widen',
        'lower_fit','lower_training_predict','lower_round','lower_widen'],inference=['query_predict']),
    'residual_calculation':dict(training=['residual','retained'],inference=[]),
    'secondary_classification':dict(training=['labels','classifier_features','classifier_names','classifier_fit',
        'calibration_probabilities','probability_widen'],inference=['query_classifier_features','query_probabilities']),
    'probabilistic_adjustment':dict(training=['offsets'],inference=['correct','final']),
    'explicit_model_state':dict(training=['learned_state'],inference=['loaded_state']),
    'identity_preserving_output':dict(training=[],inference=['domain_output']),
}


def audit(source):
    import hashlib
    triage=json.loads((ROOT/'docs/reviews/competition_nasa_pushback_source_triage.json').read_text())
    names=['Train_Models.source.py','Run_Inference.source.py','helper.py','run_model.py']
    pins={name:sha(source/name) for name in names}
    for name,digest in pins.items():
        require(digest==triage['files_sha256']['3rd Place/Phase 1/'+name],'Pinned source drift: '+name)
    tree=ast.parse((source/'Train_Models.source.py').read_text())
    initial=[];extensions=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='feature_cols' for target in node.targets):
            initial.append(ast.literal_eval(node.value))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) \
                and node.func.value.id=='feature_cols' and node.func.attr=='extend':
            extensions.append(ast.unparse(node.args[0]))
    require(initial==[['unix_time']] and set(extensions)=={'etd_features','airlinecode_features','taxitime_to_gate_features'},
        'Active source feature construction differs')
    require(len(extensions)==3,'Unexpected feature extension')
    proposed=plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed,approved=True)
        row=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",
            (SOURCE_VERSION,)).fetchone()
        snapshot=row['details']['snapshot']
        require(hashlib.sha256(_canonical(snapshot).encode()).hexdigest()==SOURCE_HASH,'Original intake snapshot drift')
        template=snapshot['template']
        stages={stage['stage_id']:stage for stage in template['stages']}
        require(set(stages)==set(REGIONS)-{'explicit_model_state','identity_preserving_output'},'Intake stage set differs')
    bases={'training':build_nasa_training_graph(),'inference':build_nasa_inference_graph()}
    for kind,base in bases.items():
        covered=[identity for region in REGIONS.values() for identity in region[kind]]
        require(len(covered)==len(set(covered)) and set(covered)=={node.node_id for node in base.nodes},
            'Lifecycle region coverage must be exact and disjoint')
    graph_coverage={}
    for key,graph in proposed['graphs'].items():
        kind=key.split('_')[0]
        base_ids={node.node_id for node in bases[kind].nodes}
        branch_bindings=[binding for binding in graph['bindings'] if binding['node_id'].startswith('p')
                         and binding['node_id'][4:] in base_ids]
        count=int(key.split('_')[1])
        require(len(branch_bindings)==count*len(base_ids),'Expanded lifecycle branch coverage differs')
        graph_coverage[key]=dict(version_id=graph['version_id'],graph_sha256=graph['graph_sha256'],
            lifecycle_node_bindings=len(branch_bindings),routing_node_bindings=len(graph['bindings'])-len(branch_bindings),
            active_populations=count,approved=True)
    corrections=[
        dict(stage='feature_engineering',disposition='correct_description',
            finding='The active source feature list extends epoch time with departure-estimate, airline-category and taxi-to-gate feature families. A distinct weather-severity feature operation is not supported by this active construction.',
            replacement='Build deterministic epoch-time, departure-estimate, saved airline-category and observation-available taxi-to-gate features, with explicit joins and training/query missing-value policies.'),
        dict(stage='primary_regression',disposition='split_stage',
            finding='The intake groups regression before residual calculation, but source lower-model fitting depends on residual-based selection. The original five-stage linear structure cannot express both fits faithfully.',
            replacement='Separate grouped internal regression, rounded residual selection and retained-population regression.'),
        dict(stage='secondary_classification',disposition='refine_contract',
            replacement='Fit strict-underestimation labels on all held-out rows using rounded training predictions; keep unrounded prediction features during query inference.'),
        dict(stage='probabilistic_adjustment',disposition='refine_contract',
            replacement='Estimate signed conditional median offsets on held-out rows, apply them only for strict probability branches, retain float32 arithmetic and final checked int32 rounding.'),
        dict(stage='all',disposition='replace_placeholder_interfaces',
            replacement='Use the explicit training-record/control to complete-model-state interface and saved-state/query-record to identity-aligned prediction-table interface.'),
    ]
    return dict(passed=True,read_only=True,catalog_mutations=0,original_intake_approved=False,
        source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,source_sha256=pins,
        original_stages=len(stages),stage_regions=REGIONS,graph_coverage=graph_coverage,
        active_feature_initialization=initial[0],active_feature_extensions=extensions,
        corrected_feature_claim_qualified=True,corrections=corrections,
        entrypoints=dict(training=proposed['graphs']['training_10']['fqdn'],
            inference={str(count):proposed['graphs'][f'inference_{count}']['fqdn'] for count in range(1,11)},
            state_handoff=dict(training_output='result',inference_input='model_states',
                contract='Complete private per-population state; native models, schema, vocabulary, calibration and target unit travel together.')),
        source_io_disposition='Timestamp-directory discovery, notebook execution, CSV intermediates and pickle files are source orchestration. The qualified reusable interface uses explicit caller-owned records and complete private state; source filesystem reproduction is not claimed.',
        remaining=['Create a corrected original-intake execution version with explicit interfaces and all required source stages; the region mapping alone does not ground the old placeholder stages.',
            'Use immutable source-version/hash provenance checks in the new qualification instead of requiring the source artifact to remain a draft forever; retain historical audits without falsifying their scope.',
            'Verify corrected intake retrieval, selection and runtime execution before approving it.'],
        validator_sha256=sha(Path(__file__)),served_evidence_sha256=sha(ROOT/'docs/reviews/nasa_population_served_verification.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.source_directory)
    (ROOT/'docs/reviews/nasa_original_intake_scope_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,original_stages=report['original_stages'],covered_approved_graphs=len(report['graph_coverage']),
        original_intake_approved=False,catalog_mutations=0)))
