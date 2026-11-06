"""Check immutable complete execution, semantic and source-branch evidence."""
import importlib
import inspect
import json
import subprocess
from pathlib import Path
from scripts.audit_decomposed_future_sales_readiness import ROOT,sha
from scripts.review_rolling_provider_semantics import packages
from scripts import stage_future_sales_corrected_intake as intake

EVIDENCE_COMMIT='039b4f4187bf25d73840276ef3aeb09e2901ab62'
EXECUTION='docs/reviews/future_sales_corrected_intake_execution.json'
SOURCE='docs/reviews/decomposed_future_sales_semantics.json'
SCOPE='docs/reviews/future_sales_original_scope.json'


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
    for key in ('cases','provider_sha256','code_sha256','nodes','edges','distinct_providers'):
        if not strict_equal(execution['execution'][key],source['execution'][key]):raise ValueError('Full execution coverage differs')


def review():
    execution=committed_report(EXECUTION);source=committed_report(SOURCE);scope=committed_report(SCOPE)
    validate_reports(execution,source,execution,source)
    plan,_=intake.qualified_plan()
    if (execution['version_id']!=plan['version_id'] or execution['graph_sha256']!=plan['graph_sha256']
        or scope['source_version_id']!=intake.ORIGINAL or scope['source_content_hash']!=intake.ORIGINAL_HASH
        or set(scope['selected_computations'])!=set(intake.CORRECTIONS) or scope['passed'] is not True):
        raise ValueError('Source and stored graph identities differ')
    for report,section in ((execution,'evidence_sha256'),(source,'software_sha256'),(source,'evidence_sha256')):
        for name,digest in report[section].items():
            path=ROOT.parent/name if name.startswith('sciona-atoms-ml/') else ROOT/name
            if sha(path)!=digest:raise ValueError('Qualified file changed: '+name)
    if sha(ROOT/'scripts/review_future_sales_original_scope.py')!=scope['reviewer_sha256']:
        raise ValueError('Source branch reviewer changed')
    for name,digest in execution['execution']['provider_sha256'].items():
        module,symbol=name.rsplit('.',1)
        fn=inspect.unwrap(getattr(importlib.import_module(module),symbol))
        if sha(Path(inspect.getfile(fn)))!=digest:raise ValueError('Qualified provider changed')
    if not strict_equal(packages(),execution['installed_packages']):raise ValueError('Qualified runtime changed')
    limits=list(dict.fromkeys(scope['limits']+source['limitations']+execution['limits']))
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=sha(ROOT/EXECUTION),
        source_review_sha256=sha(ROOT/SOURCE),source_scope_sha256=sha(ROOT/SCOPE),limitations=limits)
