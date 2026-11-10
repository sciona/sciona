"""Require immutable full stored ARC execution and complete original scope review."""
import json
import subprocess
from scripts import stage_arc_corrected_intake as intake
from scripts.plasticc_publication_evidence import strict_equal

ROOT=intake.ROOT
EVIDENCE_COMMIT='a64e3e4195c209d5094cf50a792afe77f577c78c'
EXECUTION='docs/reviews/arc_corrected_stored_execution.json'
SOURCE='docs/reviews/arc_original_scope.json'


def committed_report(path):
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+path],cwd=ROOT)
    if raw!=(ROOT/path).read_bytes():raise ValueError('Immutable qualification changed')
    return json.loads(raw)


def validate_reports(execution,source,expected_execution,expected_source):
    if not strict_equal(execution,expected_execution) or not strict_equal(source,expected_source):
        raise ValueError('Immutable qualification differs')
    if execution['passed'] is not True or source['passed'] is not True:
        raise ValueError('Passing qualification required')


def review():
    execution=committed_report(EXECUTION);source=committed_report(SOURCE)
    validate_reports(execution,source,execution,source)
    plan=intake.qualified_plan()
    if execution['version_id']!=plan['version_id'] or execution['graph_sha256']!=plan['graph_sha256']:
        raise ValueError('Executed version differs')
    if source['artifact_id']!=intake.IDENTITY or source['source_version_id']!=intake.ORIGINAL:
        raise ValueError('Reviewed original differs')
    if not strict_equal(execution['tests'],dict(tests=2,failures=0,errors=0,skipped=0)):
        raise ValueError('Two complete solver cases required')
    for field in ('stored_catalog_graph_executed','all_nodes_uncached','exact_state_edge_identity','codec_roundtrip'):
        if execution[field] is not True:raise ValueError('Incomplete stored execution evidence')
    if (execution['nodes'],execution['edges'],execution['full_solver_searches'])!=(2,1,12):
        raise ValueError('Complete source execution required')
    if len(source['source_stage_review'])!=5:raise ValueError('Five source stages required')
    for path,digest in execution['software_sha256'].items():
        if intake.sha(ROOT/path)!=digest:raise ValueError('Executed software changed')
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=intake.sha(ROOT/EXECUTION),
        source_review_sha256=intake.sha(ROOT/SOURCE),limitations=source['limitations'],
        resolved_source_corrections={r['stage_id']:r['realization'] for r in source['source_stage_review']},
        output_scope_decision=source['output_scope'],reuse_decision=source['reuse_decision'])
