"""Check reusable source raw-preparation oracle against candidate record dispatch."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
from scripts.dsb_source_preparation import SourcePreparation
from sciona.dsb_raw_inputs import _prepare_record


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    reference=SourcePreparation(source_root,pins)
    z,y,x=np.indices((24,64,64));image=np.full(z.shape,50.)
    left=((z-12)/9)**2+((y-32)/17)**2+((x-20)/9)**2<1
    right=((z-12)/9)**2+((y-32)/17)**2+((x-44)/9)**2<1
    image[left|right]=-900.
    cases=[]
    for kind,flip in [('voxel',False),('world',False),('world',True)]:
        for empty in [False,True]:
            rows=np.empty((0,4)) if empty else np.array([[20.,32.,12.,10.]])
            data=dict(volume=image,spacing=[6.,6.,6.],requested_spacing=[6.,6.,6.],annotations_xyz_diameter=rows)
            if kind=='world':data.update(left_mask=left,right_mask=right,origin_zyx=[0.,0.,0.],flip=flip)
            record=dict(kind=kind,preparation=data)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                expected=reference(record);actual=_prepare_record(record)
            for a,b in zip(actual,expected):
                np.testing.assert_array_equal(a,b)
                assert a.dtype==b.dtype
            cases.append(dict(kind=kind,flip=flip,empty_annotations=empty,exact=True))
    paths=['sciona/dsb_raw_inputs.py','sciona/dsb_training_preprocessing.py','sciona/dsb_world_preprocessing.py',
           'sciona/dsb_segmentation.py','sciona/dsb_components.py','scripts/dsb_source_preparation.py',
           'scripts/validate_dsb_source_preparation.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],cases=cases,
        adaptations=['Synthetic array IO and opaque row lookup replace files.',
            'Explicit requested resolution parameterizes the source fixed resolution for synthetic geometry.',
            'Python 2 segmentation divisions and Boolean subtraction adapted as documented.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_source_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
