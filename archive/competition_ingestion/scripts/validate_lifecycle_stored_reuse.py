"""Execute all sixteen new generic lifecycle atoms using stored catalog ports.

Fixtures describe synthetic manufacturing energy and water demand, with target
units intentionally different from the source workflow's time predictions.
"""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.guarded_prediction_graphs import graph_descriptor
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.plan_nasa_first_lifecycle_providers import ROOT,plan,sha
from scripts.validate_nasa_first_lifecycle_database_gates import check_staged


def validate():
    proposed=plan();graphs={};bindings={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed)
        for atom in proposed['atoms']:
            if atom['role']!='reusable_operation':continue
            rows=db.execute('SELECT direction,name,type_desc,constraints,required,default_value_repr,dim_signature FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                            (atom['version_id'],)).fetchall()
            def ports(direction):return [IOSpec(**{k:v for k,v in row.items() if k!='direction'}) for row in rows if row['direction']==direction]
            symbol=atom['runtime_fqdn'].split('.model_selection.')[1]
            graph=CDGExport(nodes=[AlgorithmicNode(node_id='operation',name=symbol,description='Synthetic cross-domain lifecycle check',
                concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=atom['runtime_fqdn'],inputs=ports('input'),outputs=ports('output'))],
                edges=[],metadata=dict(provider_version_id=atom['version_id'],provider_content_hash=atom['content_hash']))
            digest,nodes,edges=encode_execution_graph(graph)
            restored=decode_execution_graph(nodes,edges,digest)
            if restored!=graph:raise ValueError('Stored contract codec differs')
            graphs[symbol]=restored
            bindings[symbol]=dict(version_id=atom['version_id'],content_hash=atom['content_hash'],graph_sha256=digest)
    seen=set();cases=0
    with tempfile.TemporaryDirectory(prefix='sciona-lifecycle-reuse-') as directory:
        def execute(symbol,**inputs):
            nonlocal cases
            cases+=1;captured={}
            def capture(path,node,port,value):
                if node=='operation' and port.startswith('out_'):captured[port[4:]]=value
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-lifecycle',str(cases)).execute(inputs,cdg=graphs[symbol]))
            if result['status']!='completed' or set(captured)!={p.name for p in graphs[symbol].nodes[0].outputs}:
                raise ValueError('Stored provider output differs: '+symbol)
            seen.add(symbol);return captured

        for unit,scale,population in [('kWh',2.,'synthetic-plant'),('litre',7.,'synthetic-water-system')]:
            x=pd.DataFrame({'load':np.arange(48,dtype=float),'equipment':['A','B']*24})
            times=pd.date_range('2031-01-01',periods=48,freq='h')
            targets=np.arange(48,dtype=float)*scale
            split=execute('named_regression.temporal_split',frame=x,targets=targets,times=times,
                eligible=np.ones(48,dtype=bool),cutoff=times[36],offsets=np.full(48,scale),categorical_columns=['equipment'])
            pd.testing.assert_frame_equal(split['x_train'],x.iloc[:36])
            np.testing.assert_array_equal(split['y_valid'],targets[36:]-scale)
            assert split['categorical_indices']==[1]
            model=execute('named_regression.fit',**split,parameters=dict(iterations=6,depth=2,random_seed=4,loss_function='MAE'),early_stopping_rounds=2)['model']
            assert model.get_params()['thread_count']==1 and model.get_params()['allow_writing_files'] is False
            prediction=execute('materialized_prediction.predict',model=model,frame=x.iloc[36:])['values']
            np.testing.assert_array_equal(prediction,model.predict(x.iloc[36:],thread_count=1))
            mean=execute('materialized_prediction.average',vectors=[prediction,prediction+2*scale])['values']
            np.testing.assert_allclose(mean,prediction+scale,rtol=0,atol=1e-12)
            score=execute('fit_review.absolute_error',observed=targets[36:],predicted=mean)['score']
            assert score==float(np.abs(targets[36:]-mean).mean())
            assert execute('fit_review.accept',model=model,score=score,ceiling=score+1)['model'] is model
            importance=execute('fit_review.importance',model=model)
            np.testing.assert_array_equal(importance['values'],model.get_feature_importance(thread_count=1))
            normalized=execute('fit_review.normalize',**importance)['importance']
            assert np.isclose(normalized['importance'].sum(),1) and normalized['importance'].is_monotonic_decreasing
            bank=execute('model_banks.assemble_bank',models={'m':model},population_bindings={population:{0:'m',2:'m'}},shared_bindings={'shared':'m'})['bank']
            state=execute('native_model_state.pack_bank',bank=bank)['model_state']
            policy=dict(target_unit=unit,ordered_features=list(x.columns))
            bound=execute('policy_state.bind',model_state=state,policy=policy)['state']
            unbound=execute('policy_state.unbind',state=json.loads(json.dumps(bound)))
            assert unbound['policy']==policy and unbound['policy'] is not policy
            restored=execute('native_model_state.unpack_bank',state=unbound['model_state'])['bank'][population]
            assert restored[0] is restored[2] is restored['shared']
            np.testing.assert_array_equal(restored[0].predict(x.iloc[36:],thread_count=1),prediction)
            tables=[pd.DataFrame({'clock':[2,0],'value':[scale,2*scale]}),pd.DataFrame({'clock':[1],'value':[3*scale]})]
            merged=execute('labeled_populations.concatenate',tables=tables,labels=['line-a','line-b'],
                columns=[['clock','value']]*2,label_column='origin',order_column='clock')['table']
            assert merged['clock'].tolist()==[0,1,2] and merged['origin'].tolist()==['line-a','line-b','line-a']
            received=[]
            assert execute('graph_fallbacks.emit',value=prediction,receive=received.append)['result'] is prediction
            assert len(received)==1 and received[0] is prediction
            primary=graph_descriptor(graphs['materialized_prediction.predict'],'operation','values')
            baseline=graph_descriptor(graphs['materialized_prediction.average'],'operation','values')
            inputs=dict(model=model,frame=x.iloc[36:],vectors=[np.full(12,scale)])
            routed=execute('graph_fallbacks.predict',primary=primary,baseline=baseline,inputs=inputs,count=12,constant=3*scale)
            assert routed['route']=='primary';np.testing.assert_array_equal(routed['values'],prediction)
            routed=execute('graph_fallbacks.predict',primary=primary,baseline=baseline,inputs=dict(inputs,model=None),count=12,constant=3*scale)
            assert routed['route']=='baseline';np.testing.assert_array_equal(routed['values'],np.full(12,scale))
            routed=execute('graph_fallbacks.predict',primary=primary,baseline=baseline,inputs={},count=12,constant=3*scale)
            assert routed['route']=='constant';np.testing.assert_array_equal(routed['values'],np.full(12,3*scale))
            assert execute('graph_fallbacks.predict',primary=primary,baseline=baseline,inputs={},count=0,constant=3*scale)['route']=='empty'
    if seen!=set(graphs) or len(seen)!=16:raise ValueError('Incomplete generic provider coverage')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,stored_ports_used=True,
        provider_graphs=len(seen),executions=cases,native_fits=2,native_threads=1,target_units=['kWh','litre'],bindings=bindings,
        implementation_sha256={name:sha(ROOT/name) for name in ['scripts/validate_lifecycle_stored_reuse.py',
            'scripts/validate_nasa_first_lifecycle_database_gates.py','sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']},
        limitations=['Generic lifecycle atoms only; nineteen source-domain adapters require separate stored-contract execution.',
                     'Small synthetic native fits establish computational reuse, not empirical performance.',
                     'Production dispatch and computation are used; intermediate persistence is replaced by output capture.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_nasa_first_lifecycle_stored_reuse.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','provider_graphs','executions','native_fits','catalog_mutations']}))
