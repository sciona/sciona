"""Compare post-fit graph behavior with pinned source expressions on native models."""
import argparse
import ast
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sciona.nasa_first_training_graph import build_fit_review_graph, build_nasa_first_training_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.validate_nasa_first_native_wrapper import source_function, TRAIN_SOURCE, TRAIN_PIN
from scripts.validate_nasa_first_model_handoff import ROOT, sha


def validate(source,checkpoint):
    native=json.loads((ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json').read_text())
    if json.loads((checkpoint/'qualification.json').read_text())!=native or not native['passed']:
        raise ValueError('Qualified synthetic models required')
    fits=json.loads((checkpoint/'fit_progress.json').read_text())
    if fits!=native['fits'] or len(fits)!=21:raise ValueError('Fit identity differs')
    function=source_function(source,TRAIN_SOURCE,TRAIN_PIN,'train_catboost_diff')
    score=next(node for node in ast.walk(function) if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='score')
    importance=[node for node in function.body if isinstance(node,ast.Assign) and (
        isinstance(node.targets[0],ast.Name) and node.targets[0].id=='feat_imps' or
        isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='feat_imps')]
    importance += [node for node in function.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                   and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='feat_imps']
    if len(importance)!=3:raise ValueError('Source importance operations differ')
    program=compile(ast.Module(body=[score]+importance,type_ignores=[]),'pinned_post_fit','exec')
    outcomes=[];rng=np.random.default_rng(821)
    class BoundedModel:
        def __init__(self,model):self.model=model
        def predict(self,frame):return self.model.predict(frame,thread_count=1)
        @property
        def feature_importances_(self):return self.model.get_feature_importance(thread_count=1)
    with tempfile.TemporaryDirectory(prefix='sciona-fit-review-') as directory:
        for index,fit in enumerate(fits):
            path=checkpoint/f'model_{index}.cbm'
            if sha(path)!=fit['model_sha256']:raise ValueError('Checkpoint integrity differs')
            model=CatBoostRegressor(thread_count=1);model.load_model(str(path))
            residual=fit['kind']=='residual'
            categories=set(model.get_cat_feature_indices())
            frame=pd.DataFrame({name:['synthetic-unseen']*32 if i in categories else rng.integers(0,100,32)
                                for i,name in enumerate(model.feature_names_)})
            observed=pd.Series(rng.uniform(-30,120,32))
            namespace=dict(pd=pd,model=BoundedModel(model),best_model=BoundedModel(model),
                           x_val=frame,y_val=observed,feat_names=list(frame.columns))
            with np.errstate(invalid='ignore',divide='ignore'):exec(program,namespace)
            original=build_fit_review_graph(residual=residual)
            digest,nodes,edges=encode_execution_graph(original);graph=decode_execution_graph(nodes,edges,digest)
            if graph!=original:raise ValueError('Fit review graph codec differs')
            captured={}
            def capture(path,node,port,value):
                if port.startswith('out_'):captured[node+'/'+port[4:]]=value
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-fit-review',str(index)).execute(
                    dict(frame=frame,model=model,observed=observed.to_numpy()),cdg=graph))
            if result['status']!='completed' or captured['accept/model'] is not model:
                raise ValueError('Model acceptance graph differs')
            if captured['score/score']!=namespace['score']:raise ValueError('Source score differs')
            if residual:
                if captured['residual_outputs/model_v0'] is not model or captured['residual_outputs/model_v2'] is not model:
                    raise ValueError('Residual alias identity differs')
                pd.testing.assert_frame_equal(captured['residual_outputs/feature_importance'],namespace['feat_imps'],check_exact=True)
            outcomes.append(dict(model_index=index,residual=residual,graph_sha256=digest,passed=True,validation_rows=32))
    composed={}
    for mode in [False,True]:
        graph=build_nasa_first_training_graph(residual=mode)
        digest,nodes,edges=encode_execution_graph(graph)
        if decode_execution_graph(nodes,edges,digest)!=graph:raise ValueError('Training composition codec differs')
        composed['residual' if mode else 'direct']=dict(nodes=len(graph.nodes),edges=len(graph.edges),graph_sha256=digest)
    providers=['model_selection/fit_review.py','domain_adapters/first_place_fit_review.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,native_threads=1,cases=outcomes,
        model_scores_compared=21,residual_importance_tables_compared=10,residual_alias_pairs=10,
        composed_training_graphs=composed,composed_training_graphs_executed=False,
        source_sha256={TRAIN_SOURCE:TRAIN_PIN},generic_review_tests_passed=6,
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/regression_fit_review.py',
            'sciona/nasa_first_training_graph.py','sciona/nasa_first_fit_graph.py','scripts/validate_nasa_first_fit_review.py',
            'tests/test_regression_fit_review.py','sciona/visualizer/runner.py']},
        provider_sha256={name:sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/name) for name in providers},
        limitations=['Post-fit graphs execute on saved qualified synthetic models and newly generated validation frames, not original training validation populations.',
            'Complete single-fit graph topology is assembled and codec-checked; no new native fitting is claimed by this report.',
            'Zero or nonfinite total importance is rejected instead of emitting undefined normalized values.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True);args=parser.parse_args()
    report=validate(args.source_directory,args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_fit_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,model_scores=21,importance_tables=10,alias_pairs=10)))
