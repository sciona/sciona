"""Review three fixed image lifecycle interfaces against numeric-state evidence."""
import json
import os
import subprocess
from scripts.import_osic_image_lifecycle_drafts import reviewed_plan, ROOT, PLAN, sha
from pathlib import Path
from scripts.review_osic_archive_notices import review as archive_review
from scripts.import_flavours_provider_drafts import connect
from scripts.review_osic_runtime_integrity import review as runtime_review

DESCRIPTIONS={
    'build_image_model':'Create the fixed image model with new random weights, ready for a separate training step.',
    'export_image_prediction_state':'Save the trained model weights and their declared meanings for later predictions.',
    'restore_image_prediction_state':'Rebuild the fixed model from saved numeric weights without training it again.',
}
LIMITS=[
    'Automated Tier 3 for provisioned concrete in-process execution only; abstract witnesses reject. No human Tier 1 or empirical Tier 2 claim.',
    'The fixed B5 topology and source head remain explicit adapter policy. Generic bounded prediction remains independently reusable. Caller mappings and new-domain effectiveness require separate qualification.',
    'Caller owns feature ordering, image normalization, metadata and response meanings, coordinate declarations, random seeds, model compute limits and private artifact provenance. Coordinate equality is not evidence of semantic equivalence.',
    'Model construction consumes RNG state. State export constructs a reference model to validate topology, so it must run outside live training. Initialization does not reproduce the source notebook random stream consumed by unused architectures.',
    'Export requires a positive optimizer update counter but does not establish schedule completion. Numeric state excludes optimizer, callback and random state and cannot resume training.',
    'Export requires a new private directory; writes are not atomic and failed I/O can leave a partial directory. Caller controls trusted paths, permissions and stable files. This is not a concurrent storage protocol.',
    'Restore reconstructs only the fixed local model, checks numeric weight inventory and uses NPZ with pickle disabled. Hash and graph checks do not authenticate artifact origin. The returned model is uncompiled and has no optimizer state.',
    'Intermediate synthetic-checkpoint evidence covers exact fresh-process predictions with training and HDF5 loading disabled, plus ten manifest corruptions. It does not complete the 30-epoch image training schedule or the original CDG.',
    'Qualified installed TensorFlow, legacy Keras and archive runtime are required. Installed RECORD and archive notice checks do not establish clean installation, publisher authenticity or redistribution approval.',
    'Private model objects and local numeric state have no qualified HTTP transport or abstract simulation. Caller must bound compute threads; prediction additionally reserves one dataset worker.',
]


def review():
    intake=reviewed_plan()
    configuration={}
    for key in ('SCIONA_OSIC_DEPENDENCIES','SCIONA_OSIC_SOURCE','SCIONA_OSIC_IMAGE_STATE'):
        if not os.environ.get(key):raise ValueError('Private runtime configuration required: '+key)
        configuration[key]=Path(os.environ[key])
    notices=archive_review(configuration['SCIONA_OSIC_DEPENDENCIES'],configuration['SCIONA_OSIC_SOURCE'])
    if notices!=json.loads((ROOT/'docs/reviews/osic_archive_notices.json').read_text()):
        raise ValueError('Qualified archive notices changed')
    paths={name:ROOT/name for name in intake['implementation_sha256']}
    paths['provider_dependencies']=ROOT.parent/'sciona-atoms-ml/pyproject.toml'
    paths['provider']=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/osic_image_lifecycle.py'
    for name in ('review_osic_image_lifecycle_semantics','import_osic_image_lifecycle_drafts','plan_osic_image_lifecycle_intake',
                 'validate_osic_image_lifecycle_providers','validate_osic_image_state','review_osic_runtime_integrity','review_video_numeric_runtime','review_osic_archive_notices'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    runtime=runtime_review()
    evidence=ROOT/'docs/reviews/osic_image_lifecycle_providers.json'
    original=evidence.read_bytes()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1',TF_USE_LEGACY_KERAS='1',TF_CPP_MIN_LOG_LEVEL='3',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    run=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/validate_osic_image_lifecycle_providers.py'),
        '--dependencies',str(configuration['SCIONA_OSIC_DEPENDENCIES']),
        '--runtime-directory',str(configuration['SCIONA_OSIC_IMAGE_STATE'])],
        cwd=ROOT,env=env,capture_output=True,text=True)
    if run.returncode or evidence.read_bytes()!=original:
        raise ValueError('Connected synthetic qualification failed or changed; rerun the validator locally for diagnostics')
    execution=json.loads(original)
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        if intake['registered_provider_dependencies']:
            raise ValueError('No registered provider dependencies expected')
        for item in intake['new_providers']:
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                actual=db.execute(f'''SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',
                    (item['atom_id'],)).fetchall()
                if actual!=[dict(fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                                content_hash=item['content_hash'],trust_tier=3)]:
                    raise ValueError('Exact current provider version required')
                ports=db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(item['version_id'],)).fetchall()
                if ports!=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])):
                    raise ValueError('Stored interfaces differ')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',
                          (item['version_id'],)).fetchone():
                raise ValueError('Unexpected registered provider dependency')
            records.append(dict(artifact_id=item['atom_id'],fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                content_hash=item['content_hash'],interfaces=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])),
                operation=item['seed_fields']['description'],plain_description=DESCRIPTIONS[item['seed_fields']['source_symbol']],
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Separate fixed-model construction, inference-state export and restore support explicit graph lifecycle boundaries. Generic bounded prediction remains independent; the fixed architecture and private coordinate meanings are not generalized into a universal model-state format.',
                registered_provider_dependencies=[]))
    if before!={name:sha(path) for name,path in paths.items()} or runtime!=runtime_review() or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='osic-image-lifecycle-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,software_sha256=before,installed_packages=runtime['runtime']['installed_packages'],
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        archive_notices=notices,
        resource_policy=dict(native_threads=1,dataset_workers=1,foreground_cpu_budget=2),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/osic_image_lifecycle_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),image_training_complete=False,
                         packages=len(result['installed_packages']),approved=False)))
