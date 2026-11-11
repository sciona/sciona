"""Review the bounded Keras prediction draft for provisioned Tier 3 execution."""
import json
import os
from pathlib import Path
import subprocess

from scripts.import_keras_bounded_drafts import reviewed_plan, ROOT, PLAN, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.review_osic_runtime_integrity import review as runtime_review

DESCRIPTIONS = {
    'predict_bounded': (
        'Run a caller-provisioned legacy-Keras model with explicit native batching, ordered aligned inputs and one bounded dataset worker.',
        'Run the supplied model on batches of aligned inputs and return predictions in the same row order.'),
}
LIMITS = [
    'Automated Tier 3 for concrete provisioned legacy-Keras execution only; no human Tier 1 or empirical Tier 2 claim.',
    'Caller establishes model architecture, feature and output meanings, units, row alignment and observation availability. The operation performs no domain mapping or unit conversion.',
    'Only finite float32/float64 input arrays and one finite floating array output with matching row count are supported. Native Keras input conversion and model geometry remain in effect.',
    'Caller-provided model and custom predict code are trusted; no model loading, code sandboxing or authenticity check is provided. Custom code may have side effects; normal inference can create cached graphs.',
    'Model fitting is not invoked by the provider. Stored batch-normalization weights remain unchanged in the qualified standard-model test; this is not a state-purity guarantee for arbitrary custom models.',
    'Caller must limit model-compute threads and reserve one additional private dataset worker. Dataset autotuning is disabled; batch size alone does not bound memory or compute concurrency.',
    'Constructed motion and electrical-power examples and exact B5 batch parity establish numerical reuse, not empirical effectiveness or complete OSIC training.',
    'TensorFlow and legacy Keras must already be provisioned at the qualified versions. Installed RECORD integrity does not establish clean installation, publisher authenticity or redistribution approval.',
    'Model objects and tensors are private in-process runtime values; abstract simulation, HTTP transport and whole-CDG qualification are separate.',
]


def review():
    intake = reviewed_plan()
    provider = ROOT.parent/'sciona-atoms-ml'
    paths = {
        'provider': provider/'src/sciona/atoms/ml/keras_bounded_prediction.py',
        'implementation': ROOT/'sciona/keras_bounded_prediction.py',
        'provider_dependencies': provider/'pyproject.toml',
    }
    for name in ('review_keras_bounded_semantics', 'import_keras_bounded_drafts',
                 'plan_keras_bounded_intake', 'review_osic_runtime_integrity', 'review_video_numeric_runtime',
                 'validate_keras_bounded_provider', 'validate_keras_bounded_b5'):
        paths['scripts/'+name+'.py'] = ROOT/'scripts'/(name+'.py')
    before = {name: sha(path) for name, path in paths.items()}
    runtime = runtime_review()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1', TF_USE_LEGACY_KERAS='1', TF_CPP_MIN_LOG_LEVEL='3',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    if not env.get('SCIONA_OSIC_DEPENDENCIES'):
        raise ValueError('Private fixture dependency configuration required: SCIONA_OSIC_DEPENDENCIES')
    execution={}
    for name,arguments in [('keras_bounded_provider',[]),
            ('keras_bounded_b5',['--dependencies',env['SCIONA_OSIC_DEPENDENCIES']])]:
        path=ROOT/'docs/reviews'/(name+'.json')
        original=path.read_bytes()
        run=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/('validate_'+name+'.py')),*arguments],
            cwd=ROOT,env=env,capture_output=True,text=True)
        if run.returncode or path.read_bytes()!=original:
            raise ValueError('Fresh qualification failed or changed: '+name)
        execution[name]=json.loads(original)
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
                reuse='Reusable batching and prediction across caller-provided models and numeric input shapes. Synthetic mechanical-motion and electrical-power examples have explicit distinct application contracts; source image preparation and geometry remain separate.',
                registered_provider_dependencies=[]))
    if before != {name: sha(path) for name, path in paths.items()} or runtime != runtime_review() or intake != reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='keras-bounded-semantics.v1', passed=True, approved=False,
        proposed_tier=3, verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True,
        providers=records, connected_execution=execution, software_sha256=before,
        installed_packages=runtime['runtime']['installed_packages'], intake_plan_sha256=sha(PLAN),
        connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1, dataset_workers=1, foreground_cpu_budget=2), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/keras_bounded_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), analytic_cases=result['connected_execution']['keras_bounded_provider']['analytic_cases'],
                         packages=len(result['installed_packages']), approved=False)))
