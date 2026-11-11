"""Require immutable complete stored execution and reviewed original scope."""
import json
import subprocess
from scripts import stage_tabular_corrected_intake as intake
from scripts.plasticc_publication_evidence import strict_equal

ROOT=intake.ROOT
EVIDENCE_COMMIT='68e1b4d7abacd8e55b8b1dea28120cf4666a7454'
EXECUTION='docs/reviews/tabular_corrected_intake_execution.json'
SOURCE='docs/reviews/tabular_original_scope.json'
BASELINE='docs/reviews/complete_tabular_runtime_qualification.json'


def resolved(text):
    # Preserve the historical scope report while removing its superseded status.
    return text.replace(' Native stored-original execution and publication remain pending.', '')


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
    if execution['graph_nodes']!=4 or execution['state_edges']!=3:
        raise ValueError('Complete graph required')
    for path,digest in execution['software_sha256'].items():
        if intake.sha(ROOT/path)!=digest:raise ValueError('Qualified implementation changed')
    limits=[resolved(s['limits']) for s in source['source_stage_review']]+[
        'Only the explicitly selected source alternatives are implemented.',
        'Automated synthetic Tier 3 evidence only; no historical accuracy, generalization or human Tier 1 claim.']
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=intake.sha(ROOT/EXECUTION),
        source_review_sha256=intake.sha(ROOT/SOURCE),limitations=limits,
        resolved_source_corrections={k:resolved(v) for k,v in intake.CORRECTIONS.items()},output_scope_decision=intake.SCOPE)
