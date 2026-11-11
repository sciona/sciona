"""Review explicit population routing and injected-output contracts for Tier 3 use."""
import json
import platform

import numpy as np
import pandas as pd
from sciona.atoms.ml.domain_adapters.osic_image_routing import route_and_predict
from scripts.import_osic_image_routing_drafts import reviewed_plan,ROOT,PLAN,sha
from scripts.import_flavours_provider_drafts import connect
from scripts.validate_osic_image_routing import validate

LIMITS=[
    'Automated Tier 3 concrete in-process routing only; abstract execution rejects. No human Tier 1 or empirical Tier 2 claim.',
    'Source-specific population mapping, fixed batch size 32 and query-order policy remain explicit. Preparation, selection and prediction are injected trusted callables; no fixed registered numerical operation is hidden inside the dispatcher.',
    'Baseline and query mappings, category meanings, time and response coordinates, observation availability and compatible model inputs remain caller-owned.',
    'Finite aligned scalar predictions and ordered positions are validated. Returned vectors, positions or prediction views may borrow data. Local reference release is not a global memory bound and arbitrary callbacks may mutate inputs or retain references.',
    'Fresh synthetic tests cover routing and invalid callback outputs. The pinned population graph is supplementary prior wiring evidence over intermediate numeric state; this review does not rerun image inference or complete training.',
    'Reuse is through separate preparation, selection, prediction and reduction components, with a narrow source adapter. No claim that the source population schema or uncertainty policy transfers unchanged to other domains.',
    'No HTTP transport, serialization, concurrent-mutation safety, empirical effectiveness or original-CDG approval is established.',
]


def callback_contracts():
    baseline=pd.DataFrame(dict(entity=['b','a'],time=[3.,-1.],target=[8.,2.],numeric=[47.,53.],
        category_a=['Male','female'],category_b=['other','other'],uncertainty=[1.,2.]))
    query=pd.DataFrame(dict(key=['q0','q1'],entity=['a','b'],time=[8,-3]))
    planes={key:[object(),object(),object()] for key in ('a','b')}
    def prepare(*args):
        return dict(images=np.zeros((2,3)),metadata=np.zeros((2,4)),selected_positions=np.array([0,2],np.int64))
    def select(mapping,fields):return [mapping[k] for k in fields]
    def predict(*args):return np.zeros((2,1),np.float32)
    rejected=[]
    for label,positions in [
        ('positions_list',[0,2]),('positions_float',np.array([0.,2.])),
        ('positions_rank',np.array([[0,2]],np.int64)),('positions_short',np.array([0],np.int64)),
        ('positions_negative',np.array([-1,2],np.int64)),('positions_outside',np.array([0,3],np.int64)),
        ('positions_duplicate',np.array([0,0],np.int64)),('positions_reversed',np.array([2,0],np.int64))]:
        def malformed(*args):return dict(prepare(),selected_positions=positions)
        def forbidden(*args):raise AssertionError('Malformed positions reached a downstream operation')
        try:route_and_predict(baseline,query,planes,object(),32,malformed,forbidden,forbidden)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Malformed positions accepted')
    for label,prediction in [
        ('predictions_list',[[0.],[0.]]),('predictions_rank',np.zeros(2,np.float32)),
        ('predictions_count',np.zeros((3,1),np.float32)),('predictions_integer',np.zeros((2,1),np.int64)),
        ('predictions_nonfinite',np.full((2,1),np.nan,np.float32))]:
        def malformed(*args):return prediction
        try:route_and_predict(baseline,query,planes,object(),32,prepare,select,malformed)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Malformed predictions accepted')
    count=0
    def mixed(*args):
        nonlocal count
        count+=1
        return np.zeros((2,1),np.float32 if count==1 else np.float64)
    try:route_and_predict(baseline,query,planes,object(),32,prepare,select,mixed)
    except ValueError:rejected.append('predictions_mixed_dtype')
    else:raise AssertionError('Changing predictor dtype accepted')
    for index in range(3):
        callbacks=[prepare,select,predict];callbacks[index]=None
        try:route_and_predict(baseline,query,planes,object(),32,*callbacks)
        except ValueError:rejected.append('noncallable_'+str(index))
        else:raise AssertionError('Noncallable accepted')
    assert len(rejected)==17 and count==2
    return dict(passed=True,synthetic_only=True,rejected_contracts=rejected,
        validation_order_verified=True,consistent_prediction_dtype_required=True)


def review():
    intake=reviewed_plan()
    paths={'provider':ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/osic_image_routing.py',
           'registry':ROOT/'sciona/ghost/registry.py',
           'dispatcher':ROOT/'sciona/osic_image_routing.py'}
    for name in ('review_osic_image_routing_semantics','import_osic_image_routing_drafts',
                 'plan_osic_image_routing_intake','validate_osic_image_routing'):
        paths['scripts/'+name+'.py']=ROOT/'scripts'/(name+'.py')
    before={name:sha(path) for name,path in paths.items()}
    execution=validate()
    if execution!=json.loads((ROOT/'docs/reviews/osic_image_routing.json').read_text()):
        raise ValueError('Fresh routing qualification differs')
    contracts=callback_contracts()
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for item in intake['new_providers']:
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
            records.append(dict(artifact_id=item['atom_id'],fqdn=item['runtime_fqdn'],version_id=item['version_id'],
                content_hash=item['content_hash'],interfaces=sorted(item['interfaces'],key=lambda p:(p['direction'],p['ordinal'])),
                operation=item['seed_fields']['description'],
                plain_description='Route each mapped entity through explicit preparation, selection and prediction, then return scalar predictions and aligned coordinates for separate reduction.',
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Narrow source population adapter preserves independent reuse of injected preparation, field selection, prediction and downstream reduction. Changing application requires explicit compatible mappings, coordinates and policies; no empirical cross-domain transfer claim.',
                registered_provider_dependencies=[]))
    if before!={name:sha(path) for name,path in paths.items()} or intake!=reviewed_plan():
        raise ValueError('Qualification inputs changed during review')
    return dict(format='osic-image-routing-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=records,
        connected_execution=execution,callback_contracts=contracts,software_sha256=before,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        intake_plan_sha256=sha(PLAN),connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1,foreground_cpu_budget=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/osic_image_routing_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),approved=False)))
