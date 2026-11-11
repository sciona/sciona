"""Source AST byte-scale/PNG execution and synthetic batching boundary checks."""
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_image_boundary import batches,source_rgb,encode_real_entries


def main():
    pin=json.loads((ROOT/'docs/reviews/competition_adversarial_scipy_reference.json').read_text())
    for item in pin['files']:
        p=Path('/private/tmp/sciona_adversarial_scipy_source')/item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    s=Path('/private/tmp/sciona_adversarial_scipy_source/scipy/misc/pilutil.py').read_text()
    body=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in ('bytescale','toimage','imsave')]
    # NumPy2 removed ndarray.tostring; tobytes is its identical byte alias.
    class ByteAlias(ast.NodeTransformer):
        def visit_Attribute(self,node):
            self.generic_visit(node)
            if node.attr=='tostring':node.attr='tobytes'
            return node
    body=[ByteAlias().visit(n) for n in body]
    ns={**vars(np),'np':np,'Image':Image}
    exec(compile(ast.Module(body=body,type_ignores=[]),'<pinned-scipy-image-reference>','exec'),ns)
    cases=[np.linspace(-1,1,299*299*3,dtype=np.float32).reshape(299,299,3),
           np.linspace(-.2,.2,299*299*3,dtype=np.float32).reshape(299,299,3),
           np.full((299,299,3),.4,dtype=np.float32)]
    for x in cases:
        stream=BytesIO();ns['imsave'](stream,(x+1.)*.5,format='png')
        expected=np.array(Image.open(BytesIO(stream.getvalue())))
        np.testing.assert_array_equal(source_rgb(x),expected)
        padded=np.zeros((10,299,299,3),np.float32);padded[0]=x
        saved=encode_real_entries(padded,1)
        assert len(saved)==1
        np.testing.assert_array_equal(np.array(Image.open(BytesIO(saved[0]))),expected)
    assert source_rgb(cases[1]).min()==0 and source_rgb(cases[1]).max()==255
    assert not np.array_equal(source_rgb(cases[1]),np.round((cases[1]+1)*127.5).astype(np.uint8))
    assert np.all(source_rgb(cases[2])==0)
    input_values=(np.arange(11*299*299*3,dtype=np.uint32)%256).astype(np.uint8).reshape(11,299,299,3)
    original=input_values.copy();targets=np.arange(11,dtype=np.int64)
    results=list(batches(input_values,targets=targets))
    assert [r[0] for r in results]==[10,1]
    for start,(count,values,labels) in zip((0,10),results):
        expected=(input_values[start:start+count].astype(np.float64)/255.*2.-1.).astype(np.float32)
        np.testing.assert_array_equal(values[:count],expected)
        assert np.all(values[count:]==0) and np.all(labels[count:]==0)
        np.testing.assert_array_equal(labels[:count],targets[start:start+count])
    np.testing.assert_array_equal(input_values,original)
    # Direct float32 normalization is observably different for some byte values.
    assert np.any(results[0][1]!=(input_values[:10].astype(np.float32)/255.*2.-1.))
    paths=['sciona/adversarial_image_boundary.py','scripts/validate_adversarial_image_boundary.py',
           'docs/reviews/competition_adversarial_scipy_reference.json','docs/licenses/Adversarial-SciPy-reference-BSD.txt']
    report={'format':'adversarial-image-boundary-validation.v1','result':'passed',
            'checks':{'source_scipy_AST_and_decoded_PNG_cases':len(cases),'full_and_padded_tail_batches':2,
                      'float64_normalize_then_float32_feed':True,'padded_target_zero':True,
                      'only_real_entries_saved':True,'source_minmax_stretch_and_constant_black':True},
            'limits':'PinnedSciPy0.19.1 functions executed with current NumPy/Pillow; removed tostring alias adapted to tobytes. Decoded pixels compared, not historical PNG byte identity. Public software and synthetic arrays only. Source saved pixel minmax stretching does not preserve the normalized attack epsilon guarantee. Padded full-neural lifecycle not yet validated.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_image_boundary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
