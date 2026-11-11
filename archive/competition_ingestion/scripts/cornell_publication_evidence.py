"""Require complete immutable stored Cornell execution before publication."""
import json
import re
import subprocess
from scripts import stage_cornell_corrected_intake as intake
from scripts.plasticc_publication_evidence import strict_equal

ROOT = intake.ROOT
# Immutable full stored execution evidence.
EVIDENCE_COMMIT = '3b2f64a3ad821f9f4b8003c5968e716a1f28a5c1'
EXECUTION = 'docs/reviews/cornell_corrected_execution.json'
SOURCE = 'docs/reviews/cornell_original_scope.json'


def committed_report(path):
    if not isinstance(EVIDENCE_COMMIT, str) or not re.fullmatch('[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Committed complete stored execution is still required')
    raw = subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+path], cwd=ROOT)
    if raw != (ROOT/path).read_bytes():
        raise ValueError('Immutable qualification changed')
    return json.loads(raw)


def validate_reports(execution, source, expected_execution, expected_source):
    if not strict_equal(execution, expected_execution) or not strict_equal(source, expected_source):
        raise ValueError('Immutable qualification differs')
    flags = ('passed', 'synthetic_only', 'stored_graph_verified_before_and_after',
             'all_nodes_uncached', 'exact_state_routing', 'full_thirty_second_training',
             'ten_copy_inference', 'strict_json_output')
    if any(execution.get(key) is not True for key in flags) or source.get('passed') is not True:
        raise ValueError('Complete passing stored execution required')
    for key, value in dict(models_completed=13, training_phases=17, continuations=4,
                           actual_runner_nodes=2, catalog_mutations=0).items():
        if not strict_equal(execution.get(key), value):
            raise ValueError('Full ensemble and mutation-free evidence required')
    phases = execution.get('phases', [])
    if len(phases) != 17:
        raise ValueError('Seventeen completed phases required')
    for index, phase in enumerate(phases):
        expected = dict(epochs=2, optimizer_steps=4 if 4 <= index < 9 else 2,
                        mixup=index >= 9, selected_epoch=2)
        if not strict_equal(phase, expected):
            raise ValueError('Complete configured training and checkpoint selection required')
    if not strict_equal(execution.get('resource_policy'), dict(
            serial_models=True, torch_threads=1, interop_threads=1, native_threads=1)):
        raise ValueError('Qualified resource policy differs')
    from scripts.review_cornell_original_scope import STAGES
    stages = source.get('source_stage_review', [])
    if len(stages) != 5 or {r['stage_id'] for r in stages} != set(STAGES):
        raise ValueError('Exact five-stage source coverage required')
    if not execution.get('software_sha256'):
        raise ValueError('Executed software identities required')


def review():
    execution = committed_report(EXECUTION)
    source = committed_report(SOURCE)
    validate_reports(execution, source, execution, source)
    plan = intake.qualified_plan()
    if (execution['stored_version_id'] != plan['version_id']
            or execution['graph_sha256'] != plan['graph_sha256']
            or execution['staging_plan_sha256'] != intake.sha(intake.PLAN)):
        raise ValueError('Executed stored version or staging plan differs')
    if source['artifact_id'] != intake.IDENTITY or source['source_version_id'] != intake.ORIGINAL:
        raise ValueError('Reviewed original differs')
    for path, digest in execution['software_sha256'].items():
        if intake.sha(ROOT/path) != digest:
            raise ValueError('Executed software changed')
    return dict(review_source='automated', proposed_tier=3, verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT, execution_sha256=intake.sha(ROOT/EXECUTION),
        source_review_sha256=intake.sha(ROOT/SOURCE), limitations=source['limitations'],
        resolved_source_corrections={r['stage_id']:r['realization']+' '+r['limits']
                                    for r in source['source_stage_review']},
        reuse_decision=source['reuse_decision'])
