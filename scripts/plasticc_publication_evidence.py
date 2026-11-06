"""Check immutable complete execution, semantic and source-branch evidence."""
import importlib
import inspect
import json
import subprocess
from pathlib import Path
from scripts.audit_decomposed_plasticc_readiness import ROOT,sha
from scripts.review_decomposed_plasticc_semantics import packages
from scripts import stage_plasticc_corrected_intake as intake

EVIDENCE_COMMIT='f11d7c735f1c97cd84ab312a7444177b94a44998'
EXECUTION='docs/reviews/plasticc_corrected_intake_execution.json'
SOURCE='docs/reviews/decomposed_plasticc_semantics.json'
SCOPE='docs/reviews/decomposed_plasticc_readiness.json'
BASELINE='docs/reviews/decomposed_plasticc_graph_execution_v2.json'


def strict_equal(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return actual.keys()==expected.keys() and all(strict_equal(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return len(actual)==len(expected) and all(strict_equal(a,b) for a,b in zip(actual,expected))
    return actual==expected


def committed_report(path):
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+path],cwd=ROOT,stderr=subprocess.PIPE)
    if raw!=(ROOT/path).read_bytes():raise ValueError('Immutable qualification changed: '+path)
    return json.loads(raw)


def validate_reports(execution,source,expected_execution,expected_source):
    if not strict_equal(execution,expected_execution):raise ValueError('Stored execution qualification differs')
    if not strict_equal(source,expected_source):raise ValueError('Semantic qualification differs')
    if execution['passed'] is not True or source['passed'] is not True:raise ValueError('Passing qualification required')
    if not strict_equal(execution['execution']['cases'],source['graph_cases']):
        raise ValueError('Full execution coverage differs')
    if not strict_equal(execution['installed_packages'],source['installed_packages']):
        raise ValueError('Runtime qualification differs')


def review():
    execution=committed_report(EXECUTION);source=committed_report(SOURCE);scope=committed_report(SCOPE)
    validate_reports(execution,source,execution,source)
    baseline=committed_report(BASELINE)
    if baseline['graph_sha256']!=source['graph_digest']:
        raise ValueError('Base graph qualification differs')
    for key in ('cases','provider_sha256','implementation_sha256','nodes','edges','distinct_providers','callable_nodes','resource_policy'):
        if not strict_equal(execution['execution'][key],baseline[key]):
            raise ValueError('Full qualified execution differs: '+key)
    plan,_=intake.qualified_plan()
    if (execution['version_id']!=plan['version_id'] or execution['graph_sha256']!=plan['graph_sha256']
        or scope['source_version_id']!=intake.ORIGINAL or scope['original']['content_hash']!=intake.ORIGINAL_HASH
        or {s['stage_id'] for s in scope['source_stages']}!=set(intake.CORRECTIONS) or scope['implementation_drift']!=[]):
        raise ValueError('Source and stored graph identities differ')
    for report,section in ((execution,'evidence_sha256'),(source,'software_sha256'),(source,'evidence_sha256')):
        for name,digest in report[section].items():
            path=ROOT.parent/name if name.split('/')[0] in ('sciona-atoms','sciona-atoms-ml','sciona-atoms-signal') else ROOT/name
            if sha(path)!=digest:raise ValueError('Qualified file changed: '+name)
    for name,digest in scope['retained_evidence_sha256'].items():
        if sha(ROOT/'docs/reviews'/name)!=digest:raise ValueError('Retained source evidence changed')
    for name,digest in execution['execution']['provider_sha256'].items():
        module,symbol=name.rsplit('.',1)
        fn=inspect.unwrap(getattr(importlib.import_module(module),symbol))
        if sha(Path(inspect.getfile(fn)))!=digest:raise ValueError('Qualified provider changed')
    if not strict_equal(packages(),execution['installed_packages']):raise ValueError('Qualified runtime changed')
    # Historical pending-work statements are superseded by the exact stored
    # execution and provider-state checks; retain their immutable reports.
    limits=list(dict.fromkeys(source['limitations'][1:]+execution['limits'][1:]))
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=sha(ROOT/EXECUTION),
        source_review_sha256=sha(ROOT/SOURCE),source_scope_sha256=sha(ROOT/SCOPE),limitations=limits,
        resolved_source_corrections=intake.CORRECTIONS,output_scope_decision=intake.SCOPE)
