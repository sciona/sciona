"""Compare feature assembly with pinned source using only synthetic runtime inputs."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Collection, Dict, List, Tuple
import numpy as np
import pandas as pd
from sciona.webtraffic_assembly import assemble_features


def validate(root, source):
    pins = json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    class LegacyNumpy:
        NaN = np.nan
        def __getattr__(self, name): return getattr(np, name)
    class LegacySeries(pd.Series):
        @property
        def loc(self):
            class Lookup:
                def __getitem__(_, dates): return self.reindex(dates)
            return Lookup()
    class LegacyPandas:
        Series = LegacySeries
        def __getattr__(self, name): return getattr(pd, name)
    captured = {}
    options = SimpleNamespace(start=None, end=None, valid_threshold=0., add_days=64,
                              corr_backoffset=0, data_dir=None)
    class Parser:
        def __init__(self, **kwargs): pass
        def add_argument(self, *args, **kwargs): pass
        def parse_args(self): return options
    ns = dict(np=LegacyNumpy(), pd=LegacyPandas(), re=re, Collection=Collection,
              Dict=Dict, List=List, Tuple=Tuple, argparse=SimpleNamespace(ArgumentParser=Parser),
              VarFeeder=lambda _, tensors, plain: captured.update(tensors=tensors, plain=plain))
    names = {'find_start_end','prepare_data','single_autocorr','batch_autocorr',
             'make_page_features','uniq_page_map','encode_page_features','lag_indexes','normalize','run'}
    for filename in ['extractor.py','make_features.py']:
        path = source/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins['files'][filename]
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if (filename == 'extractor.py' and isinstance(n, ast.Assign))
                 or (isinstance(n, ast.FunctionDef) and (filename == 'extractor.py' or n.name in names))]
        for node in nodes:
            if isinstance(node, ast.FunctionDef): node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-'+filename+'>', 'exec'), ns)
        if filename == 'extractor.py': ns['extractor'] = SimpleNamespace(extract=ns['extract'])
    agents = ['all-access_spider','desktop_all-agents','mobile-web_all-agents','all-access_all-agents']
    pages = sorted(f'SyntheticAssembly{i}_{site}_{agent}' for i,site in enumerate(
        ['en.wikipedia.org','fr.wikipedia.org','commons.wikimedia.org','www.mediawiki.org']) for agent in agents)
    counts = dict(assemblies=0, tensors=0, rejected_inputs=0)
    for days in [120, 800]:
        for dtype in [np.float32, np.float64]:
            rng = np.random.RandomState(days)
            values = rng.poisson(np.arange(1,17)[:,None], size=(16,days)).astype(dtype)
            values[0] = 0; values[1,:20] = np.nan; values[2,-20:] = np.nan
            values[3,::5] = np.nan
            frame = pd.DataFrame(values,index=np.array(pages,dtype=object),columns=pd.date_range('2000-01-01',periods=days))
            original = frame.copy(deep=True)
            for threshold, future, back in [(0.,0,0),(.25,64,7),(.8,3,30)]:
                options.valid_threshold=threshold; options.add_days=future; options.corr_backoffset=back
                ns['read_x'] = lambda *args, **kwargs: frame.copy(deep=True)
                with contextlib.redirect_stdout(io.StringIO()), np.errstate(invalid='ignore', divide='ignore'):
                    ns['run']()
                    actual, plain = assemble_features(frame, threshold, future, back)
                assert plain == captured['plain']
                assert actual.keys() == captured['tensors'].keys()
                for name,a in actual.items():
                    b = captured['tensors'][name]
                    if isinstance(a,pd.DataFrame): pd.testing.assert_frame_equal(a,b)
                    elif isinstance(a,pd.Series): pd.testing.assert_series_equal(a,pd.Series(b))
                    else: np.testing.assert_equal(a,b)
                    counts['tensors'] += 1
                hits = actual['hits']
                expected = np.log1p(frame.loc[hits.index])
                pd.testing.assert_frame_equal(hits, expected)
                np.testing.assert_allclose(np.sum(actual['dow']**2,axis=1),1.,atol=1e-14)
                assert actual['lagged_ix'].shape == (days+future,4)
                assert len(actual['page_popularity']) == len(hits)
                pd.testing.assert_frame_equal(frame,original)
                counts['assemblies'] += 1
    for frame_arg, kwargs in [(frame.iloc[:,::-1],{}),(frame.iloc[::-1],{}),
                              (frame,{'add_days':-1}),(frame,{'add_days':True}),
                              (frame,{'corr_backoffset':800}),(frame,{'add_days':32768}),
                              (frame,{'valid_threshold':1.})]:
        try: assemble_features(frame_arg,**kwargs)
        except ValueError: counts['rejected_inputs']+=1
        else: raise AssertionError('Unsupported input accepted')
    paths = ['sciona/webtraffic_assembly.py','sciona/webtraffic_features.py',
             'sciona/webtraffic_pages.py','sciona/webtraffic_calendar.py',
             'scripts/validate_webtraffic_assembly.py',
             'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Original NumPy bodies run without numba; legacy NaN and calendar lookup adapted.',
                             'Source undefined/constant-feature NaNs retained; no real data or model weights.',
                             'Neural sampling, training, checkpoint averaging and inference remain unvalidated.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
