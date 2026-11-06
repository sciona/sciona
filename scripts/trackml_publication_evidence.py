"""Require immutable complete stored execution and reviewed original scope."""
import json
import subprocess
from scripts import stage_trackml_corrected_intake as intake
from scripts.review_trackml_cell_adapter_semantics import packages
from scripts.plasticc_publication_evidence import strict_equal

ROOT=intake.ROOT
EVIDENCE_COMMIT='701d27cef32ac9d402333c6d41dd4056b74a31b8'
EXECUTION='docs/reviews/trackml_corrected_intake_execution.json'
SOURCE='docs/reviews/trackml_original_scope.json'
BASELINE='docs/reviews/decomposed_trackml_cell_graph_execution.json'


def committed_report(path):
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+path],cwd=ROOT,stderr=subprocess.PIPE)
    if raw!=(ROOT/path).read_bytes():raise ValueError('Immutable qualification changed: '+path)
    return json.loads(raw)


def validate_reports(execution,source,expected_execution,expected_source):
    if not strict_equal(execution,expected_execution) or not strict_equal(source,expected_source):
        raise ValueError('Immutable qualification differs')
    if execution['passed'] is not True or source['passed'] is not True:
        raise ValueError('Passing qualification required')


def review():
    execution=committed_report(EXECUTION);source=committed_report(SOURCE)
    validate_reports(execution,source,execution,source)
    baseline=committed_report(BASELINE)
    for key,value in baseline.items():
        if key!='graph_sha256' and not strict_equal(execution['execution'][key],value):
            raise ValueError('Complete execution evidence differs: '+key)
    plan,_=intake.qualified_plan()
    if (execution['version_id']!=plan['version_id'] or execution['graph_sha256']!=plan['graph_sha256']
        or execution['artifact_id']!=plan['artifact_id'] or execution['execution']['graph_sha256']!=plan['graph_sha256']
        or not execution['stored_catalog_graph_executed'] or source['source_version_id']!=intake.ORIGINAL
        or source['source_content_hash']!=intake.ORIGINAL_HASH):
        raise ValueError('Source or executed graph identity differs')
    for hashes in (execution['evidence_sha256'],execution['execution']['implementation_sha256']):
        for path,digest in hashes.items():
            if intake.sha(ROOT/path)!=digest:raise ValueError('Qualified implementation changed')
    if not strict_equal(packages(),execution['installed_packages']):raise ValueError('Qualified runtime changed')
    limits=source['compatibility_limits'][:-1]+[s['limits'] for s in source['stages']]+[source['cell_reconstruction']['limits']]
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=intake.sha(ROOT/EXECUTION),
        source_review_sha256=intake.sha(ROOT/SOURCE),limitations=limits,
        resolved_source_corrections=intake.CORRECTIONS,output_scope_decision=intake.SCOPE)
