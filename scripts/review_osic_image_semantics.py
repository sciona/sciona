"""Review four concrete source-specific image stages against frozen evidence."""
import json
import os
import subprocess
from scripts.import_osic_image_drafts import reviewed_plan, ROOT, PLAN, sha, DEPENDENCY_SCOPES
from pathlib import Path
from scripts.review_osic_archive_notices import review as archive_review
from scripts.import_flavours_provider_drafts import connect
from scripts.plan_group_distribution_intake import check_dependency
from scripts.review_osic_runtime_integrity import review as runtime_review

DESCRIPTIONS={
    'prepare_image_training_population':'Prepare supplied images and record-level information for model training, with an explicit exclusion list.',
    'prepare_image_slice_inputs':'Prepare selected images and supplied record-level information for one model prediction group.',
    'reduce_image_predictions':'Combine image predictions with supplied starting measurements to calculate values at requested times.',
    'blend_keyed_predictions':'Match the two sets of predictions by their keys and combine them using fixed weights.',
}
LIMITS=[
    'Automated Tier 3 for provisioned concrete in-process execution only; abstract witnesses reject. No human Tier 1 or empirical Tier 2 claim.',
    'These are source-specific preparation and prediction policies. Generic affine fitting and ordered weighted summation remain independently reusable; source category meanings and weighting do not establish cross-domain effectiveness.',
    'Caller supplies mapped fields, coordinate units, ordering, complete aligned populations and actual observation availability. No private source records or exclusion identities are embedded.',
    'Training fits all entities before exclusions, uses first-row metadata and explicit rcond=-1, and can produce minimum-norm slopes for rank-deficient groups. These slopes are not uniquely identified trends.',
    'Training and inference intentionally use different image normalization and case handling. Eligible training images must decode successfully; silently shortened batches are not reproduced.',
    'Provisioned DICOM/OpenCV execution only. Disable OpenCV workers; private inputs require single-frame monochrome integer planes. Other codec/runtime configurations need separate qualification.',
    'Per-entity inference preparation retains source ordinal/count selection and ignores unselected payloads. Training preparation materializes the eligible image population; callers own its memory budget.',
    'Source median/scalar rounding and negative heuristic uncertainty are retained. The final 0.45/0.55 weighted blend is fixed source policy, not calibrated predictive uncertainty or a learned ensemble.',
    'Connected synthetic evidence uses a stable intermediate image checkpoint plus full quantile fits; it does not complete the 30-epoch image schedule or the original CDG.',
    'Private state provenance remains external. Installed RECORD and archive notice checks do not qualify redistribution, authenticate publishers or establish effectiveness.',
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
    paths['provider']=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/osic_image.py'
    for name in ('review_osic_image_semantics','import_osic_image_drafts','plan_osic_image_intake',
                 'validate_osic_image_population','review_osic_runtime_integrity','review_video_numeric_runtime','review_osic_archive_notices'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    runtime=runtime_review()
    evidence=ROOT/'docs/reviews/osic_intermediate_composition.json'
    original=evidence.read_bytes()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1',TF_USE_LEGACY_KERAS='1',TF_CPP_MIN_LOG_LEVEL='3',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    run=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/validate_osic_image_population.py'),
        '--dependencies',str(configuration['SCIONA_OSIC_DEPENDENCIES']),
        '--source-directory',str(configuration['SCIONA_OSIC_SOURCE']),
        '--state-directory',str(configuration['SCIONA_OSIC_IMAGE_STATE']),'--compose'],
        cwd=ROOT,env=env,capture_output=True,text=True)
    if run.returncode or evidence.read_bytes()!=original:
        raise ValueError('Connected synthetic qualification failed or changed; rerun the validator locally for diagnostics')
    execution=json.loads(original)
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for dependency in intake['registered_provider_dependencies']:
            check_dependency(db,dependency)
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
            dependencies=db.execute('''SELECT dependency_artifact_fqdn,dependency_content_hash,
                dependency_role,optional,port_name,binding_metadata FROM artifact_dependencies
                WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn''',(item['version_id'],)).fetchall()
            expected=[dict(dependency_artifact_fqdn=d['fqdn'],dependency_content_hash=d['content_hash'],
                dependency_role='logic_atom',optional=False,port_name='',binding_metadata=dict(
                    source_version_id=d['version_id'],scope=DEPENDENCY_SCOPES[item['seed_fields']['source_symbol']]))
                for d in item['registered_provider_dependencies']]
            if dependencies!=expected: raise ValueError('Exact dependency bindings required')
            records.append(dict(artifact_id=item['atom_id'],fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                content_hash=item['content_hash'],interfaces=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])),
                operation=item['seed_fields']['description'],plain_description=DESCRIPTIONS[item['seed_fields']['source_symbol']],
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Explicit preparation and reduction over caller-mapped inputs. Generic affine fitting and ordered summation remain reusable independently; application mappings, units, availability and empirical transfer require separate review.',
                registered_provider_dependencies=item['registered_provider_dependencies']))
    if before!={name:sha(path) for name,path in paths.items()} or runtime!=runtime_review() or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='osic-image-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,software_sha256=before,installed_packages=runtime['runtime']['installed_packages'],
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        archive_notices=notices,
        resource_policy=dict(native_threads=1,dataset_workers=1,foreground_cpu_budget=2),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/osic_image_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),image_training_complete=False,
                         packages=len(result['installed_packages']),approved=False)))
