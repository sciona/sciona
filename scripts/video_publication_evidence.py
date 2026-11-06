"""Require immutable complete stored execution and reviewed original scope."""
import json
import subprocess
from scripts import stage_video_corrected_intake as intake
from scripts.plasticc_publication_evidence import strict_equal

ROOT=intake.ROOT
EVIDENCE_COMMIT='2a8aeb5f609e6dd3bbb5436ba7f898622db77f13'
EXECUTION='docs/reviews/video_corrected_intake_execution.json'
SOURCE='docs/reviews/video_original_scope.json'
BASELINE='docs/reviews/catalog_video_multilabel_execution.json'


def committed_report(path):
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+path],cwd=ROOT,stderr=subprocess.PIPE)
    if raw!=(ROOT/path).read_bytes():raise ValueError('Immutable qualification changed: '+path)
    return json.loads(raw)


def validate_reports(execution,source,expected_execution,expected_source):
    if not strict_equal(execution,expected_execution) or not strict_equal(source,expected_source):
        raise ValueError('Immutable qualification differs')
    if execution['passed'] is not True or source['read_only'] is not True:
        raise ValueError('Passing qualification required')


def review():
    execution=committed_report(EXECUTION);source=committed_report(SOURCE)
    validate_reports(execution,source,execution,source)
    plan,_=intake.qualified_plan()  # Rechecks source-pinned runtime package closure.
    if (execution['version_id']!=plan['version_id'] or execution['graph_sha256']!=plan['graph_sha256']
        or execution['artifact_id']!=plan['artifact_id'] or not execution['stored_catalog_graph_executed']
        or source['source_version_id']!=intake.ORIGINAL or source['original']['content_hash']!=intake.ORIGINAL_HASH):
        raise ValueError('Source or executed graph identity differs')
    if not strict_equal(execution['tests'],dict(tests=3,failures=0,errors=0,skipped=0)):
        raise ValueError('Three complete stored lifecycles required')
    for field in ('all_nodes_uncached','exact_registered_bindings','full_configured_lifecycle_parity'):
        if execution[field] is not True:raise ValueError('Incomplete lifecycle evidence')
    if execution['graph_nodes']!=10 or execution['explicit_callable_edges']!=9:
        raise ValueError('Complete graph required')
    for path,digest in execution['software_sha256'].items():
        if intake.sha(ROOT/path)!=digest:raise ValueError('Qualified implementation changed')
    limits=[s['limits'] for s in source['source_stage_review']]+[
        'Only the explicitly selected source alternatives are implemented.',
        'Automated synthetic Tier 3 evidence only; no historical accuracy, generalization or human Tier 1 claim.']
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=intake.sha(ROOT/EXECUTION),
        source_review_sha256=intake.sha(ROOT/SOURCE),limitations=limits,
        resolved_source_corrections=intake.CORRECTIONS,output_scope_decision=intake.SCOPE)
