"""Check full training wiring with a recording fit; this does not claim native training."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from sciona.nasa_first_population_training_graph import build_nasa_first_population_training_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.ghost.registry import REGISTRY
import sciona.visualizer.runner as runner
from scripts.validate_nasa_first_model_handoff import ROOT,sha


def validate():
    runner._ensure_atoms_imported()
    graph=build_nasa_first_population_training_graph()
    digest,nodes,edges=encode_execution_graph(graph);restored=decode_execution_graph(nodes,edges,digest)
    if restored!=graph:raise ValueError('Population training graph codec differs')
    from sciona.atoms.ml.domain_adapters.first_place_populations import POPULATIONS
    tables={label:pd.DataFrame(dict(gufi=np.arange(32),timestamp=pd.date_range('2030-01-01',periods=32,freq='h'),
        minutes_until_pushback=np.arange(32,dtype=float)+50+i,mfs_load=np.arange(32)+i,
        etd_time_till_est_dep=np.full(32,10.),synthetic_cat_machine=['A','B']*16)) for i,label in enumerate(POPULATIONS)}
    models=[];fits=[];outputs={}
    class RecordedModel:
        def __init__(self,names):self.feature_names_=list(names)
        def predict(self,frame,*,thread_count):
            if thread_count!=1:raise ValueError('Unbounded prediction')
            return np.ones(len(frame))
        def get_feature_importance(self,*,thread_count):
            if thread_count!=1:raise ValueError('Unbounded importance')
            return np.arange(1,len(self.feature_names_)+1,dtype=float)
    def recording_fit(x_train,y_train,x_valid,y_valid,categorical_indices,parameters,early_stopping_rounds):
        if parameters['n_estimators']!=20000 or parameters['min_data_in_leaf']!=5000 or early_stopping_rounds!=60:
            raise ValueError('Source fit policy differs')
        model=RecordedModel(x_train.columns);models.append(model)
        fits.append((model,x_train.copy(),y_train.copy(),x_valid.copy(),y_valid.copy(),categorical_indices))
        return model
    runtime='sciona.atoms.ml.model_selection.named_regression.fit'
    registration=dict(REGISTRY[runtime],impl=recording_fit)
    def capture(path,node,port,value):
        if port.startswith('out_'):outputs[node+'/'+port[4:]]=value
    with tempfile.TemporaryDirectory(prefix='sciona-training-wiring-') as directory:
        with patch.dict(REGISTRY,{runtime:registration}),patch.object(runner,'RUNS_DIR',Path(directory)), \
                patch.object(runner,'save_intermediate_value',side_effect=capture), \
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-training-wiring','case').execute(
                dict(population_tables=tables,target_name='minutes_until_pushback',end_train='2030-01-01 16:00:00'),cdg=restored))
    if result['status']!='completed' or len(fits)!=21 or len(result['trace'])!=len(graph.nodes):
        raise ValueError('Incomplete source training topology execution')
    bank=outputs['bank/bank']
    for index,label in enumerate(POPULATIONS):
        slots=bank[label.upper()]
        if slots[0] is not slots[2] or slots[0] is not outputs[f'residual{index}/fit/model'] or slots[1] is not outputs[f'direct{index}/fit/model']:
            raise ValueError('Local fit/alias wiring differs')
        if slots['global_model'] is not outputs['direct10/fit/model']:
            raise ValueError('Shared global fit wiring differs')
        for mode in ['residual','direct']:
            frame=outputs[f'{mode}{index}/split/x_train'];targets=outputs[f'{mode}{index}/split/y_train']
            expected=tables[label].iloc[:16]
            np.testing.assert_array_equal(frame['mfs_load'],expected['mfs_load'])
            np.testing.assert_array_equal(targets,expected['minutes_until_pushback']-(10 if mode=='residual' else 0))
    if len({id(model) for slots in bank.values() for model in slots.values()})!=21:
        raise ValueError('Independent model identities differ')
    if len(outputs['direct10/split/x_train'])!=160 or len(outputs['direct10/split/x_valid'])!=160:
        raise ValueError('Global population membership differs')
    providers=['domain_adapters/first_place_populations.py','model_selection/labeled_populations.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,recording_fit_backend=True,
        native_graph_training_qualified=False,nodes=len(graph.nodes),edges=len(graph.edges),graph_sha256=digest,
        fit_invocations=21,independent_models=21,local_slots=30,shared_global_models=1,residual_alias_pairs=10,
        importance_outputs=10,global_training_rows=160,global_validation_rows=160,
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/nasa_first_population_training_graph.py',
            'sciona/nasa_first_training_graph.py','sciona/nasa_first_fit_graph.py','scripts/validate_nasa_first_population_training_wiring.py']},
        provider_sha256={name:sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/name) for name in providers},
        limitations=['Only the native fit implementation is replaced by a recording model; split, validation arithmetic, importance normalization, global concatenation and model-bank operations execute normally.',
            'Source-sized native graph training, raw training input composition and catalog approval remain pending.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_nasa_first_population_training_wiring.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nodes=report['nodes'],edges=report['edges'],fit_invocations=21,native_graph_training_qualified=False)))
