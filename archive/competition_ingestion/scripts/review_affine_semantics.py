"""Review reusable affine fitting for provisioned automated Tier 3 publication."""
import inspect
import json
from pathlib import Path

import numpy as np
from sciona.atoms.numerical.statistics import affine_fit as atoms
from sciona.ghost import registry,abstract,dimensions
from scripts.import_affine_drafts import reviewed_plan,ROOT,PLAN,sha
from scripts.import_flavours_provider_drafts import connect
from scripts.review_group_distribution_semantics import packages
from scripts.validate_affine_provider import validate

LIMITS=[
    'Automated Tier 3 for concrete provisioned in-process execution only; no human Tier 1 or empirical Tier 2 claim.',
    'Caller establishes units, alignment and observation availability. No centering, weighting, regularization or temporal filtering.',
    'Rank below two returns minimum-norm coefficients, not uniquely identified slope/intercept. Empty residual arrays do not imply zero error.',
    'Explicit cutoff is runtime dependent: None uses the installed NumPy size-dependent default; -1 uses machine precision.',
    'Float64 design conversion can lose large-integer precision. Abstract witness intentionally rejects; no static dimension inference is qualified.',
    'Synthetic motion and sensor examples establish numerical reuse only; the original OSIC workflow is not approved by this review.',
    'Installed RECORD hashes and active requirements are checked; independent distribution provenance and other platforms remain outside qualification.'
]


def review():
    intake=reviewed_plan()
    paths={module.__name__:Path(inspect.getfile(module)) for module in (atoms,registry,abstract,dimensions)}
    for name in ('review_affine_semantics','review_group_distribution_semantics','validate_affine_provider',
                 'import_affine_drafts','plan_affine_intake'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/ (name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    installed=packages()
    execution=validate()
    if execution!=json.loads((ROOT/'docs/reviews/affine_provider_qualification.json').read_text()):
        raise ValueError('Independent numerical qualification changed')
    dtype_cases=0
    for dtype in (np.int32,np.int64,np.float32,np.float64):
        for n in (1,2,7):
            x=np.arange(n,dtype=dtype);y=(3*x+2).astype(dtype)
            result=atoms.affine_least_squares(x,y,-1)
            assert result['coefficients'].dtype==np.float64
            assert result['singular_values'].shape==(min(n,2),)
            assert result['residual_sums_of_squares'].shape==((1,) if n>2 else (0,))
            assert result['rank']==min(n,2)
            dtype_cases+=1
    boundary=1.+np.arange(8)*4e-16
    assert atoms.affine_least_squares(boundary,np.arange(8.),-1)['rank']==2
    assert atoms.affine_least_squares(boundary,np.arange(8.),None)['rank']==1
    repeated=atoms.affine_least_squares(np.array([2.,2.,2.]),np.array([4.,6.,8.]),-1)
    assert repeated['rank']==1 and repeated['residual_sums_of_squares'].size==0
    np.testing.assert_allclose(repeated['coefficients'],[2.4,1.2],rtol=1e-13,atol=1e-13)
    try:
        atoms.witness_affine_least_squares()
    except ValueError:
        pass
    else:
        raise ValueError('Unsupported abstract execution accepted')
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for item in intake['new_providers']:
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                actual=db.execute(f'''SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',
                    (item['atom_id'],)).fetchall()
                if actual!=[dict(fqdn=item['runtime_fqdn'],version_id=item['version_id'],content_hash=item['content_hash'],trust_tier=3)]:
                    raise ValueError('Exact current provider version required')
                ports=db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(item['version_id'],)).fetchall()
                if ports!=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])):
                    raise ValueError('Stored interfaces differ')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(item['version_id'],)).fetchone():
                raise ValueError('Unexpected registered atom dependency')
            records.append(dict(artifact_id=item['atom_id'],fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                content_hash=item['content_hash'],interfaces=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])),
                operation='Fit uncentered affine least squares with explicit singular-value rank cutoff and minimum-norm diagnostics.',
                plain_description='Fit a straight line to paired numbers and report whether its slope and starting value can be determined uniquely.',
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Shared numerical fitting for motion measurements and sensor calibration; application units, observation selection and metadata interpretation remain separate.',
                registered_provider_dependencies=[]))
    if before!={name:sha(path) for name,path in paths.items()} or installed!=packages() or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='affine-semantics.v1',passed=True,approved=False,proposed_tier=3,verdict='acceptable_with_limits',
        catalog_mutations=0,synthetic_only=True,providers=records,independent_numerical_cases=128,
        dtype_shape_cases=dtype_cases,rank_cutoff_boundary_verified=True,repeated_coordinate_minimum_norm=True,
        abstract_rejected=True,software_sha256=before,installed_packages=installed,intake_plan_sha256=sha(PLAN),
        connected_execution_commit=intake['execution_evidence_commit'],connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1,workers=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/affine_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),dtype_cases=result['dtype_shape_cases'],
        packages=len(result['installed_packages']),approved=False)))
