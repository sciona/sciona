"""Complete pinned-source tracking scheduler parity on in-memory synthetic events."""
import argparse
import asyncio
import tempfile
from unittest.mock import patch
import copy
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import pprint
import re
from string import ascii_lowercase
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sciona.trackml_detector import discover_detector
from sciona.trackml_graph import build_trackml_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
import sciona.atoms.physics.trackml_execution as provider
from sciona.trackml_intersections import Intersector
from sciona.trackml_lifecycle import TrackLifecycle
from sciona.trackml_execution import execute_tracking, check_layer_population, TrackMLInputError
from sciona.trackml_payload import encode_table, decode_table, encode_layer_functions, execute_payload, GridLayerFunctions
from scripts.validate_trackml_detector import synthetic_modules
from scripts.validate_trackml_extension import ExtensionMaps
from scripts.validate_trackml_cells import LegacySeries
from scripts.validate_trackml_layer_integration import source_namespace
from scripts.validate_trackml_neighbors import synthetic_hits,LegacyFrame


class LegacyEventFrame(LegacyFrame):
    @property
    def _constructor(self):return LegacyEventFrame
    @property
    def _constructor_sliced(self):return LegacySeries

class LegacyPandas:
    DataFrame=LegacyEventFrame
    def __getattr__(self,name):return getattr(pd,name)


def validate(root,source):
    ns=source_namespace(root,source);ns.update(re=re,pprint=pprint,ascii_lowercase=ascii_lowercase,pd=LegacyPandas())
    graph_hash,node_rows,edge_rows=encode_execution_graph(build_trackml_graph())
    graph=decode_execution_graph(node_rows,edge_rows,graph_hash)
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    for name in ['geometry.py','candidates.py','cells.py','algorithm.py']:
        path=source/'trackml_solution'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/'+name]
        tree=ast.parse(path.read_text())
        if name=='geometry.py':nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        elif name in ('candidates.py','cells.py'):nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        else:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Algorithm')
            cls.bases=[];cls.name='OriginalLifecycle';nodes=[cls]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<original-'+name+'>','exec'),ns)
    raw_find_tracks=ns['OriginalLifecycle'].findTracks
    # Reference compatibility scope: two empty-population stops, no numerical edits.
    source_tree=ast.parse((source/'trackml_solution/algorithm.py').read_text())
    original_cls=next(n for n in source_tree.body if isinstance(n,ast.ClassDef) and n.name=='Algorithm')
    fn=next(n for n in original_cls.body if isinstance(n,ast.FunctionDef) and n.name=='findTracks')
    class EmptyStops(ast.NodeTransformer):
        def visit_Assert(self,node):
            if ast.unparse(node.test)=='secnh is not None':
                return ast.parse('if secnh is None or secnh.empty:\n    submission_df_parts.append(Candidates(event_test).submit(fill=False))\n    break').body[0]
            return node

        def visit_For(self,node):
            self.generic_visit(node)
            if ast.unparse(node.iter)=="range(int(run.params['follow__niter']))":
                node.body.insert(0,ast.parse('if run.candidates.n == 0: break').body[0])
            return node
        def visit_With(self,node):
            self.generic_visit(node)
            if ast.unparse(node.items[0].context_expr)=="self.timed('evaluating')":
                return [ast.parse('if run.candidates.n == 0: break').body[0],node]
            return node
    exec(compile(ast.fix_missing_locations(ast.Module(body=[EmptyStops().visit(fn)],type_ignores=[])),
                 '<source-with-empty-stops>','exec'),ns)
    ns['OriginalLifecycle'].findTracks=ns['findTracks']
    modules=synthetic_modules(scale=10)
    modules['module_id']=np.arange(1,len(modules)+1)
    modules['pitch_u']=.1;modules['pitch_v']=.2;modules['module_t']=.3
    for i,dst in enumerate('xyz'):
        for j,src in enumerate('uvw'):modules[f'rot_{dst}{src}']=float(i==j)
    with threadpool_limits(limits=1):
        spec=discover_detector(modules);ospec=ns['DetectorSpec'](SimpleNamespace(detectors_df=LegacyEventFrame(modules)))
    counts=dict(event_cases=0,commit_rounds=0,assigned_hits=0,fill_hits=0,postprocess_cases=0,raw_source_empty_failures=0,raw_source_no_seed_failures=0,cell_cases=0,map_cases=0,cell_filter_calls=0,cell_ranking_calls=0,assignments_by_mode={},execution_cases=0,sparse_rejections=0,payload_cases=0,payload_rejections=0,serialized_graph_cases=0)
    class Event:
        def __init__(self,hits,legacy=False,cells=False):
            self.hits_df=LegacyEventFrame(hits) if legacy else hits.copy()
            self.max_hit_id=len(hits);self.event_id=1
            self.has_cells=cells;self.has_truth=False;self.opened=0;self.closed=0
            self.xyz=np.full((len(hits)+1,3),np.nan)
            self.xyz[1:]=hits[['x','y','z']].to_numpy()
            self.modules=np.r_[0,hits.module_id.to_numpy()]
            cell_rows=[]
            for hit in hits.hit_id:
                for pixel in range(1+int(hit)%3):
                    cell_rows.append(dict(hit_id=hit,ch0=pixel,ch1=pixel%2,value=1.+pixel*.2))
            self.cells_df=pd.DataFrame(cell_rows)
        def hitCoordinatesById(self,ids):return tuple(self.xyz[np.asarray(ids)].T.copy())
        def hitModuleIdById(self,ids):return self.modules[np.asarray(ids)] if self.has_cells else np.asarray(ids)%19
        def open(self):self.opened+=1
        def close(self):self.closed+=1
        def summary(self):return 'synthetic event'
    def instrument(algo):
        outputs=[];rounds=[];fits=[]
        algo.createOrAppendSubmissionFile=lambda filename,frame,append=False:outputs.append(frame.copy(deep=True))
        setup=algo.setupRun
        def call(*args,**kwargs):
            run=setup(*args,**kwargs)
            rounds.append((kwargs['i_commit'],kwargs['sunset'],run.available_hits_fraction))
            return run
        algo.setupRun=call
        fitting=algo.fitTracks
        def fit(*args,**kwargs):
            result=fitting(*args,**kwargs)
            c=args[0].candidates
            fits.append((c._candidates.copy(),c._fit.copy()))
            return result
        algo.fitTracks=fit
        filtering=algo.dropNeighborsInconsistentWithCellFeatures
        def filter_cells(*args,**kwargs):
            counts['cell_filter_calls']+=1
            return filtering(*args,**kwargs)
        algo.dropNeighborsInconsistentWithCellFeatures=filter_cells
        ranking=algo.evaluateCellFeatureConsistency
        def rank_cells(*args,**kwargs):
            counts['cell_ranking_calls']+=1
            return ranking(*args,**kwargs)
        algo.evaluateCellFeatureConsistency=rank_cells
        return outputs,rounds,fits
    for seed in range(2):
        hits=synthetic_hits(modules,2031+seed)
        hits.loc[hits.volume_id==8,'z']*=.1
        hits['module_id']=np.tile(modules.module_id.to_numpy(),12)
        for mode in ['single','multi','sunset','odd','maps','cells','cells_maps','no_seed','empty_seed']:
            params=ns['OriginalLifecycle'].default_params.copy()
            params.update({'follow__niter':2,'nb__dist_threshold':10.,'pair__dist_threshold':10.,
                           'pair__diff_threshold':10.,'rank__ntop_qu':1.,'rank__ntop_li':1.,
                           'rank__ntop':1000,'follow__drop_start':0,
                           'commit__niter':2 if mode=='multi' else 1,
                           'commit__nmax':5 if mode=='multi' else 1000,
                           'post__nonphys_odd':mode=='odd'})
            if mode=='no_seed':params['nb__nlayers']=0
            if mode=='empty_seed':params.update(nb__cyl_origin_area=0.,nb__cap_origin_radius=0.)
            if mode=='sunset':params['sunset__ADD__nb__dist_threshold']=1.
            maps=ExtensionMaps() if 'maps' in mode else None
            cells='cells' in mode
            if cells:params['nb__cells_cut']=1.
            a=TrackLifecycle(spec,Intersector(spec),params,layer_functions=maps)
            b=ns['OriginalLifecycle'].__new__(ns['OriginalLifecycle'])
            b.spec=ospec;b.params=params.copy();b.layer_functions=maps;b.intersector=ns['Intersector'](ospec)
            b.log=lambda *args:None;b.timed=lambda *args:nullcontext();b.indent=nullcontext()
            probe=copy.copy(b);probe.createOrAppendSubmissionFile=lambda *args,**kwargs:None
            with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                try:
                    raw_find_tracks(probe,None,[Event(hits,legacy=True,cells=cells)],submission_filename='in-memory',
                                    analysis=False,score_intermediate=False,score_final=False)
                except AssertionError:
                    assert mode=='no_seed'
                    counts['raw_source_no_seed_failures']+=1
                except ValueError as error:
                    assert 'zero-size array to reduction operation maximum' in str(error)
                    counts['raw_source_empty_failures']+=1
            ao,ar,af=instrument(a);bo,br,bf=instrument(b)
            ae=Event(hits,cells=cells);be=Event(hits,legacy=True,cells=cells)
            with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                av=a.findTracks(None,[ae],submission_filename='in-memory',analysis=False,score_intermediate=False,score_final=False)
                bv=b.findTracks(None,[be],submission_filename='in-memory',analysis=False,score_intermediate=False,score_final=False)
            assert av==bv==[] and len(ao)==len(bo)==1
            pd.testing.assert_frame_equal(pd.DataFrame(ao[0]),pd.DataFrame(bo[0]));assert ar==br and len(af)==len(bf)
            for actual,expected in zip(af,bf):
                for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
            result=ao[0]
            if mode in ('single','maps','cells','cells_maps','no_seed','empty_seed'):
                runtime_hits=hits.copy(deep=True)
                if not cells:runtime_hits['module_id']=runtime_hits.hit_id.to_numpy()%19
                saved=runtime_hits.copy(deep=True)
                with threadpool_limits(limits=1):
                    executed=execute_tracking(modules,runtime_hits,cells=ae.cells_df if cells else None,
                                              layer_functions=maps,params=params,event_id=1)
                pd.testing.assert_frame_equal(executed,pd.DataFrame(result))
                pd.testing.assert_frame_equal(runtime_hits,saved)
                counts['execution_cases']+=1
                payload=dict(version=1,modules=encode_table(modules),hits=encode_table(runtime_hits),
                             cells=encode_table(ae.cells_df) if cells else None,
                             layer_functions=encode_layer_functions(maps,cylinder_count=3,cap_count=4) if maps else None,
                             params=params,event_id=1)
                serialized=json.dumps(payload,allow_nan=False)
                wire=json.loads(serialized)
                with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                    packed=execute_payload(wire)
                pd.testing.assert_frame_equal(decode_table(packed['assignments']),executed)
                assert json.dumps(wire,allow_nan=False)==serialized
                counts['payload_cases']+=1
                if mode in ('single','cells_maps'):
                    captured={}
                    def capture(directory,node,name,value):
                        if name.startswith('out_'):captured[(node,name[4:])]=value
                    with tempfile.TemporaryDirectory(prefix='synthetic-trackml-graph-') as directory:
                        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture),threadpool_limits(limits=1):
                            outcome=asyncio.run(runner.CDGExecutionSession(None,'synthetic-trackml','case').execute({'payload':wire},cdg=graph))
                    assert outcome['status']=='completed',outcome.get('error','Graph failed')
                    assert captured[('encode','result')]==packed
                    counts['serialized_graph_cases']+=1



            assert result.hit_id.is_unique
            np.testing.assert_array_equal(np.sort(result.hit_id),hits.hit_id)
            assert ae.opened==ae.closed==be.opened==be.closed==1
            if mode in ('no_seed','empty_seed'):assert (result.track_id==0).all()
            counts['event_cases']+=1;counts['commit_rounds']+=len(ar)
            counts['assigned_hits']+=int((result.track_id>0).sum());counts['fill_hits']+=int((result.track_id==0).sum())
            counts['postprocess_cases']+=int(mode=='odd')
            counts['cell_cases']+=int(cells);counts['map_cases']+=int(maps is not None)
            counts['assignments_by_mode'][mode]=counts['assignments_by_mode'].get(mode,0)+int((result.track_id>0).sum())
    for sparse in [hits.loc[hits.volume_id!=8],hits.iloc[:0],hits.groupby(['volume_id','layer_id']).head(1)]:
        try:check_layer_population(spec,sparse,4)
        except TrackMLInputError:counts['sparse_rejections']+=1
        else:raise AssertionError('Unsupported layer population was accepted')
    for bad in [
        {'columns':[{'name':'a','dtype':'int8','values':[128]}]},
        {'columns':[{'name':'a','dtype':'int32','values':[1.5]}]},
        {'columns':[{'name':'a','dtype':'float64','values':[None]}]},
        {'columns':[{'name':'a','dtype':'object','values':[1]}]},
    ]:
        try:decode_table(bad)
        except TrackMLInputError:counts['payload_rejections']+=1
        else:raise AssertionError('Invalid numeric table was accepted')
    # Missing grids and missing values have explicit JSON representations.
    grid=dict(functions=['dp0'],grids=[dict(is_cylinder=True,layer_id=0,function='dp0',
        axes=[[0.,1.]],values=[None,2.])])
    restored=GridLayerFunctions(json.loads(json.dumps(grid,allow_nan=False)))
    assert np.isnan(restored.getGridValues(is_cylinder=True,layer_id=0,functions=['dp0'])[0][1][0])
    assert restored.getGridValues(is_cylinder=False,layer_id=0,functions=['dp0'])==[(None,None)]
    bad=copy.deepcopy(grid);bad['grids'].append(copy.deepcopy(bad['grids'][0]))
    try:GridLayerFunctions(bad)
    except TrackMLInputError:counts['payload_rejections']+=1
    else:raise AssertionError('Duplicate grid accepted')
    assert counts['assigned_hits']>0 and counts['commit_rounds']>counts['event_cases']
    # Bind the complete local tracking implementation chain used by this gate.
    paths=[str(p) for p in sorted((root/'sciona').glob('trackml_*.py'))]
    paths=[str(Path(p).relative_to(root)) for p in paths]
    paths+=['scripts/validate_trackml_extension.py','scripts/validate_trackml_cells.py','scripts/validate_trackml_lifecycle.py','scripts/validate_trackml_neighbors.py',
            'scripts/validate_trackml_layer_integration.py','scripts/validate_trackml_detector.py',
            'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        serialized_graph_sha256=graph_hash,graph_nodes=len(node_rows),graph_edges=len(edge_rows),
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Synthetic cell-free/cell-enabled events and complete analytic runtime maps; no learned calibration claim.',
                     'Runtime parameters selected for synthetic event size; no competition accuracy claim.',
                     'Outputs captured in memory; ground-truth scoring and diagnostic/file output paths excluded.',
                     'Explicit empty-candidate and no-seed stops added to candidate and source reference; raw source failures counted separately. Sparse/exhausted layer behavior remains.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
