"""Review exact differentiable loss drafts for provisioned Tier 3 execution."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from scripts.import_distribution_loss_drafts import reviewed_plan, ROOT, PLAN, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.review_osic_runtime_integrity import review as runtime_review

DESCRIPTIONS = {
    'mean_pinball': (
        'Compute the mean asymmetric quantile loss over aligned rows and explicitly ordered quantile columns.',
        'Score quantile predictions with caller-chosen penalties for underprediction and overprediction.'),
    'clipped_laplace_score': (
        'Compute a mean clipped error-and-spread score with explicit scale floor and absolute-error cap.',
        'Score point predictions and their spread using caller-chosen limits on error and minimum spread.'),
}
LIMITS = [
    'Automated Tier 3 for concrete provisioned TensorFlow execution only; no human Tier 1 or empirical Tier 2 claim.',
    'Caller establishes target units, row identity, column alignment and observation availability; no domain policy is inferred.',
    'Pinball carries target units. The clipped score logarithm depends on the numerical unit convention; scores from different unit conventions cannot be compared directly.',
    'Crossed predictions are permitted. Neither objective establishes calibrated uncertainty or ordered quantiles.',
    'GradientTape must enclose the call; JSON transport, remote execution and abstract simulation do not preserve this qualified differentiable interface.',
    'Finite float32 inputs may overflow intermediate arithmetic. Callers must check loss and gradients before optimizer updates.',
    'Synthetic analytic tests and source comparisons qualify numerical behavior, not empirical transfer or the full OSIC workflow.',
    'Installed RECORD integrity and dependency constraints do not authenticate publishers or qualify clean-platform installation or redistribution.',
]


def review():
    intake = reviewed_plan()
    provider = ROOT.parent/'sciona-atoms-ml'
    tests = provider/'tests/test_tensorflow_distribution_losses_provider.py'
    paths = {
        'provider': provider/'src/sciona/atoms/ml/tensorflow_distribution_losses.py',
        'provider_tests': tests,
        'provider_dependencies': provider/'pyproject.toml',
    }
    for name in ('review_distribution_loss_semantics', 'import_distribution_loss_drafts',
                 'plan_distribution_loss_intake', 'review_osic_runtime_integrity', 'review_video_numeric_runtime'):
        paths['scripts/'+name+'.py'] = ROOT/'scripts'/(name+'.py')
    before = {name: sha(path) for name, path in paths.items()}
    runtime = runtime_review()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1', TF_USE_LEGACY_KERAS='1', TF_CPP_MIN_LOG_LEVEL='3',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='sciona-distribution-loss-review-') as temporary:
        report = Path(temporary)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', '--disable-warnings',
            str(tests), '--junitxml='+str(report)], cwd=ROOT, env=env, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Synthetic provider tests failed: '+run.stdout[-4000:]+run.stderr[-1000:])
        suites = list(ET.parse(report).getroot().iter('testsuite'))
        counts = {key: sum(int(s.get(key, '0')) for s in suites)
                  for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=5, failures=0, errors=0, skipped=0):
        raise ValueError('All five analytic provider tests required')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for item in intake['new_providers']:
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                actual = db.execute(f'''SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',
                    (item['atom_id'],)).fetchall()
                if actual != [dict(fqdn=item['runtime_fqdn'], version_id=item['version_id'],
                                   content_hash=item['content_hash'], trust_tier=3)]:
                    raise ValueError('Exact current provider version required')
                ports = db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (item['version_id'],)).fetchall()
                if ports != sorted(item['interfaces'], key=lambda p: (p['direction'], p['ordinal'])):
                    raise ValueError('Stored interfaces differ')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (item['version_id'],)).fetchone():
                raise ValueError('Unexpected registered atom dependency')
            operation, plain = DESCRIPTIONS[item['seed_fields']['source_symbol']]
            records.append(dict(artifact_id=item['atom_id'], fqdn=item['runtime_fqdn'], version_id=item['version_id'],
                content_hash=item['content_hash'], interfaces=sorted(item['interfaces'], key=lambda p: (p['direction'],p['ordinal'])),
                operation=operation, plain_description=plain,
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Continuous prediction objectives with explicit caller units and policy; synthetic duration and energy numerical examples. Application feature construction, model architecture, optimizer and calibration assessment remain separate.',
                registered_provider_dependencies=[]))
    if before != {name: sha(path) for name, path in paths.items()} or runtime != runtime_review() or intake != reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='distribution-loss-semantics.v1', passed=True, approved=False,
        proposed_tier=3, verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True,
        providers=records, tests=counts, software_sha256=before,
        installed_packages=runtime['runtime']['installed_packages'], intake_plan_sha256=sha(PLAN),
        connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1, workers=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/distribution_loss_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=result['tests'],
                         packages=len(result['installed_packages']), approved=False)))
