"""Review five concrete source-specific quantile stages against frozen evidence."""
import json
import os
import subprocess
from scripts.import_osic_quantile_drafts import reviewed_plan, ROOT, PLAN, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.plan_group_distribution_intake import check_dependency
from scripts.review_osic_runtime_integrity import review as runtime_review

DESCRIPTIONS={
    'assemble_quantile_populations':'Join the supplied training, baseline and query tables, removing all repeated training keys.',
    'fit_joint_quantile_features':'Build shared numeric features from the complete supplied population, including queries.',
    'train_prepared_quantile_folds':'Train two fixed prediction models from prepared numbers and retain their saved states.',
    'replay_prepared_quantile_folds':'Use both saved models to predict from prepared numbers without training again.',
    'reduce_quantile_predictions':'Average the model outputs and apply supplied known outcomes before combining prediction branches.',
}
LIMITS=[
    'Automated Tier 3 for provisioned concrete in-process execution only; abstract witnesses reject. No human Tier 1 or empirical Tier 2 claim.',
    'Source-specific adapters and a fixed model recipe; generic loss atoms remain independently reusable. No empirical cross-domain effectiveness is established.',
    'Caller supplies mapped table fields, units, alignment, available observations and complete query population. Labels do not prove coordinate equivalence.',
    'Joint preparation is transductive: queries can change training features. Fold validation does not establish a leakage-free estimate.',
    'Training uses the fixed source architecture, 600 epochs per fold and explicit caller seeds. It changes process random-generator state and must execute serially.',
    'Saved states are private inference-only numeric artifacts. Authenticity is external; no exact training resume or arbitrary architecture loading.',
    'Crossed predictions and negative raw uncertainty are retained. Known-outcome overrides require actual availability before prediction.',
    'Six synthetic regression fits establish connected behavior against the preceding qualified lifecycle, not historical random draws or predictive accuracy.',
    'The complete image branch, final combined graph and original OSIC intake remain separate pending work.',
    'Installed RECORD and active dependency checks do not authenticate publishers, certify redistribution or qualify other platforms.',
]


def review():
    intake=reviewed_plan()
    paths={name:ROOT/name for name in intake['implementation_sha256']}
    paths['provider']=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/osic_quantile.py'
    for name in ('review_osic_quantile_semantics','import_osic_quantile_drafts','plan_osic_quantile_intake',
                 'validate_osic_training_boundary','review_osic_runtime_integrity','review_video_numeric_runtime'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    runtime=runtime_review()
    evidence=ROOT/'docs/reviews/osic_training_boundary.json'
    original=evidence.read_bytes()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1',TF_USE_LEGACY_KERAS='1',TF_CPP_MIN_LOG_LEVEL='3',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    run=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/validate_osic_training_boundary.py')],
        cwd=ROOT,env=env,capture_output=True,text=True)
    if run.returncode or evidence.read_bytes()!=original:
        raise ValueError('Connected qualification failed or changed: '+run.stdout[-1000:]+run.stderr[-1000:])
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
                    source_version_id=d['version_id'],scope='Fixed model construction and training objective; inference reconstructs the compiled model without fitting.'))
                for d in item['registered_provider_dependencies']]
            if dependencies!=expected: raise ValueError('Exact dependency bindings required')
            records.append(dict(artifact_id=item['atom_id'],fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                content_hash=item['content_hash'],interfaces=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])),
                operation=item['seed_fields']['description'],plain_description=DESCRIPTIONS[item['seed_fields']['source_symbol']],
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Explicit source recipe over caller-mapped tables and prepared numeric inputs. Application mapping and availability policies stay outside the generic loss atoms; empirical transfer requires separate evaluation.',
                registered_provider_dependencies=item['registered_provider_dependencies']))
    if before!={name:sha(path) for name,path in paths.items()} or runtime!=runtime_review() or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='osic-quantile-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,software_sha256=before,installed_packages=runtime['runtime']['installed_packages'],
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1,workers=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/osic_quantile_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),full_schedule_fits=6,
                         packages=len(result['installed_packages']),approved=False)))
