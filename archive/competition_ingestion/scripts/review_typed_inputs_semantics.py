"""Review corrected generic input conversions for concrete Tier 3 use."""
import json
import platform

import numpy as np
from scripts.stage_typed_inputs_correction import reviewed_plan,ROOT,PLAN,sha,RUNNER
from uuid import UUID,uuid5
from sciona.atoms.ml.pipeline import typed_inputs as provider
from scripts.import_flavours_provider_drafts import connect
from scripts.validate_typed_inputs_unicode_correction import validate

LIMITS=[
    'Automated Tier 3 concrete in-process conversion only; abstract execution rejects. No human Tier 1 or empirical Tier 2 claim.',
    'Casting requires a nonempty plain native float32/float64 ndarray and an explicit float32 or float64 string. Always return independent storage; preserve shape and element order.',
    'Native narrowing rounds and can underflow to signed zero. Nonfinite inputs and outputs including overflow reject. Caller decides acceptable precision loss; no normalization, clipping or unit conversion.',
    'Descriptor encoding requires unique ordered nonempty lists of nonblank Unicode scalar-value strings. Reject surrogate code points, including explicit UTF-16 pairs, to preserve exact standard JSON round trips.',
    'Preserve descriptor/component order and spelling without trimming or Unicode normalization. Compact ASCII-escaped JSON avoids delimiter collisions but does not infer semantics or array-column alignment.',
    'Caller owns coordinates, axis meanings, availability, feature alignment and private label metadata. Synthetic motion/power and Unicode fixtures establish constructed reuse, not empirical effectiveness.',
    'Fresh tests cover conversions and Unicode correction. Pinned quantile-graph evidence is supplementary prior composition evidence; this review does not retrain folds or rerun the graph.',
    'Historical encoder failure remains attached to its superseded version. No HTTP transport, concurrent-mutation safety or original-CDG approval claim.',
]
PLAIN={
    'cast_finite_array':'Copy the supplied numbers into the requested floating-point format while keeping their shape and order.',
    'encode_feature_descriptors':'Turn each structured feature label into an unambiguous text string while keeping its parts and order.',
}


def precision_checks():
    values=np.array([1+2**-24,1+3*2**-24,2**-150,-2**-150,-0.0],dtype=np.float64)
    result=provider.cast_finite_array(values,'float32')
    np.testing.assert_array_equal(result,np.array([1.,1+2**-22,0.,-0.,-0.],np.float32))
    np.testing.assert_array_equal(np.signbit(result),[False,False,False,True,True])
    scalar=provider.cast_finite_array(np.array(1.25,np.float32),'float64')
    assert scalar.shape==() and scalar.dtype==np.float64 and scalar.item()==1.25
    cube=np.arange(24,dtype=np.float64).reshape(2,3,4).transpose(2,0,1)
    copied=provider.cast_finite_array(cube,'float64')
    np.testing.assert_array_equal(copied,cube)
    assert copied.shape==cube.shape and not np.shares_memory(copied,cube)
    return dict(passed=True,round_to_even_verified=True,signed_underflow_verified=True,
                negative_zero_preserved=True,scalar_shape_preserved=True,strided_tensor_copy_verified=True)


def review():
    intake=reviewed_plan()
    paths={'provider':ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/pipeline/typed_inputs.py',
           'registry':ROOT/'sciona/ghost/registry.py'}
    for name in ('review_typed_inputs_semantics','stage_typed_inputs_correction',
                 'plan_typed_inputs_correction','validate_typed_inputs_provider','validate_typed_inputs_unicode_correction'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    execution=validate()
    if execution!=json.loads((ROOT/'docs/reviews/typed_inputs_unicode_correction.json').read_text()):
        raise ValueError('Fresh routing qualification differs')
    precision=precision_checks()
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for item in intake['providers']:
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                rows=db.execute(f'''SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                    WHERE a.{key}=%s AND v.is_latest''',(item['atom_id'],)).fetchall()
                if rows!=[dict(fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                               content_hash=item['content_hash'],trust_tier=3)]:
                    raise ValueError('Exact current provider version required')
                ports=db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(item['version_id'],)).fetchall()
                if ports!=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])):
                    raise ValueError('Stored interface differs')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(item['version_id'],)).fetchone():
                raise ValueError('No registered atom dependencies expected')
            if item['new_version_required']:
                prior=item['previous']
                for prefix in ('atom','artifact'):
                    old=db.execute(f'SELECT content_hash,is_latest FROM {prefix}_versions WHERE version_id=%s',(prior['version_id'],)).fetchone()
                    if old!=dict(content_hash=prior['content_hash'],is_latest=False):raise ValueError('Superseded version changed')
                failure=db.execute('SELECT passed,status,version_id::text FROM artifact_audit_evidence WHERE evidence_id=%s',
                    (uuid5(UUID(prior['version_id']),RUNNER+'-counterexample'),)).fetchone()
                if failure!=dict(passed=False,status='completed',version_id=prior['version_id']):raise ValueError('Historical encoder failure missing')
            records.append(dict(artifact_id=item['atom_id'],fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                content_hash=item['content_hash'],interfaces=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])),
                operation=item['seed_fields']['description'],
                plain_description=PLAIN[item['seed_fields']['source_symbol']],
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Application-independent conversion with explicit dtype or structured labels. Constructed motion/power use preserves order without embedding source policy; precision, coordinate meaning and feature alignment remain caller-owned.',
                registered_provider_dependencies=[]))
    if before!={name:sha(path) for name,path in paths.items()} or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='typed-inputs-semantics.v2',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,precision_checks=precision,software_sha256=before,
        runtime=dict(python=platform.python_version(),numpy=np.__version__),
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['evidence_commit'],
        connected_execution_sha256=intake['evidence_sha256'],
        resource_policy=dict(native_threads=1,foreground_cpu_budget=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/typed_inputs_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),approved=False)))
