"""Pinned-source page feature checks using invented page strings only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Collection,Dict
import numpy as np
import pandas as pd
from sciona import webtraffic_pages as candidate


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    class LegacyNumpy:
        NaN=np.nan
        def __getattr__(self,name):return getattr(np,name)
    ns=dict(np=LegacyNumpy(),pd=pd,re=re,Collection=Collection,Dict=Dict)
    for filename in ['extractor.py','make_features.py']:
        path=source/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files'][filename]
        tree=ast.parse(path.read_text())
        if filename=='extractor.py':nodes=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef))]
        else:nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'make_page_features','uniq_page_map','encode_page_features'}]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-'+filename+'>','exec'),ns)
        if filename=='extractor.py':ns['extractor']=SimpleNamespace(extract=ns['extract'])
    agents=['all-access_spider','desktop_all-agents','mobile-web_all-agents','all-access_all-agents']
    counts=dict(extractions=0,groupings=0,encodings=0,constant_encodings=0)
    for seed in range(8):
        rng=np.random.RandomState(2231+seed);strings=[]
        for i,site in enumerate(['en.wikipedia.org','fr.wikipedia.org','commons.wikimedia.org','www.mediawiki.org']):
            for j in range(3):
                term=f'SyntheticPage{i}_{j}' if i<2 else f'Category:SyntheticPage{i}_{j}'
                for agent in agents:
                    if rng.uniform()>.2:strings.append(term+'_'+site+'_'+agent)
        pages=np.array(sorted(strings),dtype=object)
        for series in [False,True]:
            values=pd.Series(pages) if series else pages
            a=candidate.extract_page_fields(values);b=ns['extract'](values)
            pd.testing.assert_frame_equal(a,b);counts['extractions']+=1
            assert set(a.site)=={'wikipedia.org','commons.wikimedia.org','www.mediawiki.org'}
            wiki=a.site=='wikipedia.org'
            assert a.loc[wiki,'marker'].isna().all()
            assert (a.loc[~wiki,'marker']=='Category').all()
            assert a.loc[~wiki,'country'].isna().all()
        grouped=candidate.uniq_page_map(pages)
        np.testing.assert_array_equal(grouped,ns['uniq_page_map'](pages))
        expected={}
        for i,page in enumerate(pages):
            suffix=next(a for a in agents if page.endswith('_'+a))
            base=page[:-len(suffix)-1]
            expected.setdefault(base,[-1]*4)[agents.index(suffix)]=i
        np.testing.assert_array_equal(grouped,np.array(list(expected.values()),dtype=np.int32))
        counts['groupings']+=1
        af=candidate.make_page_features(pages);bf=ns['make_page_features'](pages)
        pd.testing.assert_frame_equal(af,bf)
        ae=candidate.encode_page_features(af);be=ns['encode_page_features'](bf)
        assert ae.keys()==be.keys()
        for name in ae:
            pd.testing.assert_frame_equal(ae[name],be[name])
            np.testing.assert_allclose(ae[name].mean(),0.,atol=1e-14)
            np.testing.assert_allclose(ae[name].std(ddof=1),1.,atol=1e-14)
            counts['encodings']+=1
        # Source divides by sample std; a single-category population stays NaN.
        constant=pd.DataFrame({'agent':['desktop_all-agents']*3})
        a=candidate.encode_page_features(constant);b=ns['encode_page_features'](constant)
        pd.testing.assert_frame_equal(a['agent'],b['agent']);assert a['agent'].isna().all().all()
        counts['constant_encodings']+=1
    paths=['sciona/webtraffic_pages.py','scripts/validate_webtraffic_pages.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Sorted runtime object-string arrays required for source grouping; no real page names used.',
                     'One-hot features use sample standard deviation; constant categories produce source NaNs.',
                     'Full feature assembly, neural training and final ensemble remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_pages.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
