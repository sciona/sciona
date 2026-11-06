"""Build a reviewable provider dependency plan including deferred runtime objects."""
import json
from pathlib import Path
from scripts.inventory_dsb_decomposed_providers import inventory, ROOT, sha

DL = 'sciona.atoms.dl.'
NUM = 'sciona.atoms.numerical.'
SPATIAL = DL + 'detection.dsb_spatial_recipes.'
DEFERRED = [
    (DL+'detection.dsb_decomposed_training.dsb_decomposed_training_batches',
     [SPATIAL+n for n in ('crop_detection','augment_detection','assign_anchor_labels','crop_classifier_proposal','augment_classifier')]
     +[NUM+'statistics.threshold_sampling.sequential_softmax_indices'],
     ['sciona/dsb_decomposed_batches.py','sciona/dsb_decomposed_detector_samples.py','sciona/dsb_decomposed_sampling.py'],
     'Returned TrainingBatchFactory executes source sampling and spatial recipes when consumed.'),
    (DL+'detection.dsb_evaluation_sampling.validation_batches',
     [SPATIAL+n for n in ('crop_detection','assign_anchor_labels','crop_classifier_proposal')],
     ['sciona/dsb_decomposed_validation_batches.py','sciona/dsb_decomposed_evaluation_samples.py'],
     'Returned ValidationBatchFactory executes detector and classifier crop recipes when consumed.'),
    (DL+'detection.dsb_pooling_runtime.restore_shared_training_runtime',
     [DL+'probability_aggregation.noisy_or_with_baseline',DL+'spatial.centered_window.centered_window'],
     ['sciona/dsb_decomposed_model_runtime.py','sciona/dsb_decomposed_network.py'],
     'Returned shared classifier calls generic center-window extraction and noisy-OR during forward passes.'),
    (DL+'detection.dsb_pooling_runtime.restore_inference_models',
     [DL+'probability_aggregation.noisy_or_with_baseline',DL+'spatial.centered_window.centered_window'],
     ['sciona/dsb_decomposed_model_runtime.py','sciona/dsb_decomposed_network.py'],
     'Returned inference classifier calls generic center-window extraction and noisy-OR during forward passes.'),
]


def plan():
    current = inventory(offline=True)
    trace_path = ROOT/'docs/reviews/dsb_provider_runtime_dependencies.json'
    trace = json.loads(trace_path.read_text())
    if (not trace['passed'] or trace['graph_digest'] != current['graph_digest']
            or trace['inventory_sha256'] != sha(ROOT/'docs/reviews/dsb_decomposed_provider_inventory.json')):
        raise ValueError('Runtime dependency evidence drift')
    qualification_path = ROOT/'docs/reviews/dsb_lifecycle_contract_graph_source_parity.json'
    qualification = json.loads(qualification_path.read_text())
    records = {r['runtime_fqdn']:r for r in current['providers']}
    edges = {(e['caller'],e['callee']):dict(kind='observed_runtime_call', source_review=[])
             for e in trace['dependencies']}
    for caller, callees, paths, reason in DEFERRED:
        sources = {}
        for name in paths:
            digest = sha(ROOT/name)
            if qualification['implementation_sha256'][name] != digest:
                raise ValueError('Deferred dependency source drift')
            sources[name] = digest
        for callee in callees:
            edge = edges.setdefault((caller,callee),dict(kind='deferred_object_dependency',source_review=[]))
            edge['source_review'].append(dict(reason=reason,sources=sources))
    result = []
    for (caller,callee),evidence in sorted(edges.items()):
        if caller not in records or callee not in records or caller==callee:
            raise ValueError('Unknown or self-dependent provider')
        result.append(dict(caller=caller,callee=callee,
            caller_version=records[caller]['version_id'],callee_version=records[callee]['version_id'],
            caller_content_hash=records[caller]['content_hash'],callee_content_hash=records[callee]['content_hash'],
            **evidence))
    return dict(approved=False,catalog_mutations=0,graph_digest=current['graph_digest'],
        providers=len(records),dependency_links=len(result),observed_links=len(trace['dependencies']),
        deferred_links=sum(e['kind']=='deferred_object_dependency' for e in result),
        complete_static_closure=False,dependencies=result,trace_sha256=sha(trace_path),
        planner_sha256=sha(Path(__file__)),qualification_sha256=sha(qualification_path),
        remaining_review=['Check source helpers and conditional paths for additional registered dependencies.',
            'Review provider interfaces and external library/version constraints before staging.',
            'Arbitrary caller callbacks are outside this qualification; publication must constrain their use.'])


if __name__=='__main__':
    report=plan()
    (ROOT/'docs/reviews/dsb_provider_dependency_plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('providers','dependency_links','observed_links','deferred_links','complete_static_closure')}))
