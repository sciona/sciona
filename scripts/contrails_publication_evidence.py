"""Require both complete stored Contrails variants before original publication."""
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
                 'strict_json_output', 'source_sized_inputs_and_outputs', 'stored_graph_verified_before_and_after'):
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
    required = {'scripts/validate_contrails_corrected_execution.py',
                'scripts/stage_contrails_corrected_intake.py', 'sciona/catalog_contrails_graph.py',
                'scripts/review_contrails_augmentation_runtime.py',
                'docs/reviews/contrails_augmentation_runtime.json',
                'sciona/contrails_lifecycle.py', 'sciona/contrails_runtime.py'}
    if not required <= set(hashes):
        raise ValueError('Numerical and augmentation execution identities required')


# Pin only after both stored execution reports have completed and been committed.
EVIDENCE_COMMIT = None
SOURCE = 'docs/reviews/contrails_original_scope.json'


def committed_report(path):
    commit = EVIDENCE_COMMIT
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Full immutable stored execution commit required')
    raw = subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)
    if raw != (ROOT/path).read_bytes():
        raise ValueError('Committed execution evidence changed')
    return json.loads(raw)


def validate_stored_reports(reports, plan):
    if set(reports) != set(FOLDS):
        raise ValueError('Both complete stored variants required')
    for variant, report in reports.items():
        validate_report(report, variant)
        for key, expected in [('stored_version_id', plan['version_id']),
                              ('graph_sha256', plan['graph_sha256']),
                              ('staging_plan_sha256', plan['plan_sha256'])]:
            if not strict_equal(report.get(key), expected):
                raise ValueError('Stored execution identity differs: '+key)


def review():
    from scripts import stage_contrails_corrected_intake as intake
    from scripts.validate_contrails_corrected_execution import prerequisites
    paths = {v:'docs/reviews/contrails_corrected_execution_'+v+'.json' for v in FOLDS}
    reports = {v:committed_report(path) for v,path in paths.items()}
    source = committed_report(SOURCE)
    plan = intake.qualified_plan()
    validate_stored_reports(reports, dict(plan, plan_sha256=sha(intake.PLAN)))
    if source['artifact_id'] != intake.IDENTITY or source['source_version_id'] != intake.ORIGINAL:
        raise ValueError('Reviewed original differs')
    for report in reports.values():
        for path,digest in report['software_sha256'].items():
            if sha(ROOT/path) != digest:
                raise ValueError('Executed software changed')
        if report['prerequisite_sha256'] != sha(ROOT/'docs/reviews/contrails_current_contracts.json'):
            raise ValueError('Component qualification changed')
    prerequisites()
    return dict(review_source='automated', proposed_tier=3, verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,
        execution_sha256={v:sha(ROOT/path) for v,path in paths.items()},
        source_review_sha256=sha(ROOT/SOURCE), limitations=source['limitations'],
        resolved_source_corrections={r['stage_id']:r['realization']+' '+r['correction']
                                    for r in source['source_stage_review']},
        reuse_decision=source['reuse_decision'])
