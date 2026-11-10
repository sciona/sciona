"""Reject partial or unbounded Contrails evidence before provider promotion."""
import json
import subprocess
from scripts.audit_contrails_original_readiness import ROOT, sha
from scripts.plasticc_publication_evidence import strict_equal

FOLDS = {'v43': [('temporal',5), ('temporal',7), ('single',3), ('single',4)],
         'v47': [('temporal',5), ('temporal',6), ('temporal',7), ('temporal',8), ('temporal',9),
                 ('single',0), ('single',2), ('single',3), ('single',4), ('single',5)]}
RESOURCE_POLICY = dict(serial_models=True, torch_threads=1, interop_threads=1,
                       native_threads=1, data_loader_workers=0, opencv_threads=1, opencv_opencl=False)


def validate_report(report, variant):
    expected = FOLDS[variant]
    for name in ('passed', 'synthetic_only', 'all_nodes_uncached', 'exact_state_routing',
                 'strict_json_output', 'source_sized_inputs_and_outputs'):
        if report.get(name) is not True:
            raise ValueError('Complete passing execution required: '+name)
    for name, value in dict(variant=variant, approved=False, catalog_mutations=0,
                            actual_runner_nodes=2, full_models_completed=len(expected),
                            optimizer_updates=3*len(expected), resource_policy=RESOURCE_POLICY).items():
        if not strict_equal(report.get(name), value):
            raise ValueError('Incomplete execution or resource policy: '+name)
    wanted = [dict(branch=branch, fold=fold, epochs=1, optimizer_updates=3, terminal_checkpoint=True)
              for branch,fold in expected]
    if not strict_equal(report.get('folds'), wanted):
        raise ValueError('Every exact source fold and terminal checkpoint is required')
    hashes = report.get('software_sha256', {})
    required = {'scripts/validate_contrails_current_graph.py',
                'scripts/review_contrails_augmentation_runtime.py',
                'docs/reviews/contrails_augmentation_runtime.json',
                'sciona/contrails_lifecycle.py', 'sciona/contrails_runtime.py'}
    if not required <= set(hashes):
        raise ValueError('Numerical and augmentation execution identities required')


def review(commit):
    """Require both committed variants and unchanged numerical/augmentation code."""
    if not isinstance(commit,str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Full immutable evidence commit required')
    results = {}
    graph_hashes = set()
    for variant in FOLDS:
        path = 'docs/reviews/contrails_current_graph_'+variant+'.json'
        raw = subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
        if raw != (ROOT/path).read_bytes():
            raise ValueError('Committed execution evidence changed')
        report = json.loads(raw)
        validate_report(report,variant)
        for name,digest in report['software_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('Executed software changed')
        if report['prerequisite_sha256'] != sha(ROOT/'docs/reviews/contrails_current_contracts.json'):
            raise ValueError('Component qualification changed')
        graph_hashes.add(report['graph_sha256'])
        results[variant] = dict(report_sha256=sha(ROOT/path), models=report['full_models_completed'])
    if len(graph_hashes) != 1:
        raise ValueError('Both variants must use the same reviewed graph')
    from scripts.validate_contrails_current_graph import prerequisites
    prerequisites()
    return dict(passed=True, approved=False, catalog_mutations=0, evidence_commit=commit,
                variants=results, graph_sha256=next(iter(graph_hashes)),
                scope='Both full local variants; corrected-original stored execution remains separate.')
