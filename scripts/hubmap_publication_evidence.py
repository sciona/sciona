"""Require committed complete stored HuBMAP execution before Tier 3 publication."""
import json
import re
import subprocess
from scripts.audit_hubmap_original_readiness import ROOT,sha
from scripts import hubmap_execution_evidence as lifecycle
from scripts.plasticc_publication_evidence import strict_equal

EVIDENCE_COMMIT=None
EXECUTION='docs/reviews/hubmap_corrected_execution.json'
SOURCE='docs/reviews/hubmap_original_scope.json'
REQUIRED_SOFTWARE=lifecycle.REQUIRED_SOFTWARE|{
    'scripts/validate_hubmap_corrected_execution.py',
    'scripts/stage_hubmap_corrected_intake.py','sciona/catalog_hubmap_graph.py'}


def validate_report(report):
    for name,value in dict(format='hubmap-corrected-execution.v1',
            stored_graph_verified_before_and_after=True).items():
        if not strict_equal(report.get(name),value):
            raise ValueError('Complete stored execution required: '+name)
    # Share the full eight-fold, twenty-epoch, routing and resource requirements.
    # Changing the schema label here does not bypass any computational checks.
    lifecycle.validate_report(dict(report,format='hubmap-current-graph.v2'))
    if not REQUIRED_SOFTWARE<=set(report['software_sha256']):
        raise ValueError('Stored graph software identities required')
    if not isinstance(report.get('staging_commit'),str) or not re.fullmatch('[0-9a-f]{40}',report['staging_commit']):
        raise ValueError('Full immutable staging commit required')
    if not isinstance(report.get('staging_plan_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',report['staging_plan_sha256']):
        raise ValueError('Exact staging plan fingerprint required')


def validate_stored_report(report,plan):
    validate_report(report)
    for key,expected in [('stored_version_id',plan['version_id']),
                         ('graph_sha256',plan['graph_sha256']),
                         ('staging_plan_sha256',plan['plan_sha256'])]:
        if not strict_equal(report.get(key),expected):
            raise ValueError('Stored execution identity differs: '+key)


def committed_report(path):
    if not isinstance(EVIDENCE_COMMIT,str) or not re.fullmatch('[0-9a-f]{40}',EVIDENCE_COMMIT):
        raise ValueError('Full immutable stored execution commit required')
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+path],cwd=ROOT)
    if raw!=(ROOT/path).read_bytes():raise ValueError('Committed qualification changed')
    return json.loads(raw)


def qualification_limits(source,providers,report):
    groups=[source['limitations'],providers['limitations'],report['limits']]
    if any(not isinstance(group,list) or not group or
           any(not isinstance(item,str) or not item.strip() for item in group) for group in groups):
        raise ValueError('Explicit limits required from every qualification layer')
    return list(dict.fromkeys(item for group in groups for item in group))


def review():
    report=committed_report(EXECUTION)
    source=committed_report(SOURCE)
    providers=committed_report('docs/reviews/hubmap_provider_contracts.json')
    from scripts import stage_hubmap_corrected_intake as intake
    from scripts import validate_hubmap_corrected_execution as execution
    plan=execution.qualified_staging()
    validate_stored_report(report,dict(plan,plan_sha256=sha(intake.PLAN)))
    if report['staging_commit']!=execution.STAGING_COMMIT:
        raise ValueError('Executed staging commit differs')
    if source['artifact_id']!=intake.IDENTITY or source['source_version_id']!=intake.ORIGINAL:
        raise ValueError('Reviewed original differs')
    for path,digest in report['software_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Executed software changed')
    if sha(ROOT/'docs/reviews/hubmap_current_contracts.json')!=report['prerequisite_sha256']:
        raise ValueError('Component qualification changed')
    execution.prerequisites()
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=sha(ROOT/EXECUTION),
        source_review_sha256=sha(ROOT/SOURCE),limitations=qualification_limits(source,providers,report),
        resolved_source_corrections={r['stage_id']:r['realization']+' '+r['correction']
                                    for r in source['source_stage_review']},
        reuse_decision=source['reuse_decision'])
