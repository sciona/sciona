"""Execute the pinned image inference cells using only synthetic raw inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from sciona.osic_image_inference import predict,select_slice_positions,prepare_slice_inputs
from sciona.osic_predictions import image_trajectories
from scripts.review_osic_original_scope import ROOT,INFERENCE,REPOSITORY,REVISION

PINS={6:'56eee99ee8632632a72876cf33e6234c7becfc0f2f8ce3d7e86c8f0a9987a3e7',
      9:'4ff759ede08f197111d0aada45ef7800a0960e6982fb76143b42f377fccdf621',
      15:'b8931415e14679cf6fc9bcdb52a1f3f222a776f514f0791629e13c097cdedf17',
      17:'b2b3f38aa020fc2eaddb3ced944b05cbe458114678e399ddf5a319ef129619e3'}


def validate(directory):
    cv2.setNumThreads(0)
    assert cv2.getNumThreads()==1
    notebook=json.loads((Path(directory)/INFERENCE).read_text())
    code={}
    for index,digest in PINS.items():
        tree=ast.parse(''.join(notebook['cells'][index]['source']))
        assert hashlib.sha256(ast.unparse(tree).encode()).hexdigest()==digest
        code[index]=compile(tree,'<pinned-synthetic-image-inference>','exec')
    baselines=pd.DataFrame(dict(entity=['synthetic-a','synthetic-b'],time=[1.,-2.],target=[7.,11.],
        numeric=[47.,61.],category_a=['Male','male'],category_b=['other','Ex-smoker'],uncertainty=[.1,3.]))
    queries=pd.DataFrame(dict(key=['synthetic-b_5','synthetic-a_-2','synthetic-b_0','synthetic-a_3'],
        entity=['synthetic-b','synthetic-a','synthetic-b','synthetic-a'],time=np.array([5,-2,0,3],dtype=np.int64)))
    labels=[[-1,0,1,2,3,4,5,9,10,11],[4,1,9,0,3]]
    planes={entity:[(str(label)+'.dcm',(np.arange(13*19).reshape(13,19)+i*11-100).astype('int16'))
                    for i,label in enumerate(labels[g])]
            for g,entity in enumerate(baselines['entity'])}
    source_baselines=baselines.rename(columns=dict(entity='Patient',time='Weeks',target='FVC',
        numeric='Age',category_a='Sex',category_b='SmokingStatus',uncertainty='Percent'))
    source_queries=pd.DataFrame(dict(Patient_Week=queries['key'],FVC=np.zeros(4),Confidence=np.zeros(4)))
    raw={(entity,name):plane for entity,records in planes.items() for name,plane in records}
    cases=[]
    for case in ('synthetic_linear_predictor','float32_median_rounding','float32_scalar_rounding'):
        calls=[]
        def predictor(inputs):
            image,metadata=inputs
            calls.append((image.copy(),metadata.copy()))
            if case=='synthetic_linear_predictor':
                return (image.mean(axis=(1,2,3))+.125*metadata[:,0]+metadata[:,1]).astype('float32')[:,None]
            following=np.nextafter(np.float32(1),np.float32(2))
            output=np.full((len(image),1),following,dtype=np.float32)
            if case=='float32_median_rounding':
                output[:len(output)//2]=1.
            return output
        scope=dict(np=np,cv2=cv2,models=[SimpleNamespace(predict=predictor)],
            pd=SimpleNamespace(read_csv=lambda path:source_queries.copy() if 'sample_submission' in path else source_baselines.copy()),
            os=SimpleNamespace(listdir=lambda path:[name for name,_ in planes[path.rstrip('/').split('/')[-1]]]),
            pydicom=SimpleNamespace(dcmread=lambda path:SimpleNamespace(pixel_array=raw[tuple(path.split('/')[-2:])])) )
        for index in (6,9,15,17):
            exec(code[index],scope)
        source_calls=calls[:]
        calls.clear()
        actual=predict(baselines,queries,planes,predictor)
        assert actual['keys']==scope['sub']['Patient_Week'].tolist()
        np.testing.assert_array_equal(actual['values'],scope['sub'][['FVC','Confidence']].to_numpy())
        assert len(calls)==len(source_calls)==2
        for pair1,pair2 in zip(calls,source_calls):
            for a,b in zip(pair1,pair2):
                assert a.dtype==b.dtype
                np.testing.assert_array_equal(a,b)
        calls.clear()
        prepared=[]
        for row in baselines.itertuples():
            prepared.append(prepare_slice_inputs(planes[row.entity],row.numeric,row.category_a,row.category_b))
        assert not calls
        for item,expected_inputs in zip(prepared,source_calls):
            for array,expected_array in zip((item['images'],item['metadata']),expected_inputs):
                assert array.dtype==expected_array.dtype
                np.testing.assert_array_equal(array,expected_array)
        slice_predictions=[predictor([item['images'],item['metadata']]) for item in prepared]
        reduced=image_trajectories(slice_predictions,baselines['time'].to_numpy(),baselines['target'].to_numpy(),
            baselines['uncertainty'].to_numpy(),np.array([1,0,1,0],dtype=np.int64),queries['time'].to_numpy())
        np.testing.assert_array_equal(reduced,actual['values'])
        np.testing.assert_array_equal(actual['selected_positions'][0],np.arange(1,9))
        np.testing.assert_array_equal(actual['selected_positions'][1],[0,1,3,4])
        cases.append(dict(case=case,queries=4,selected_slices=12,exact_predictor_inputs=True,
                          exact_source_outputs=True,query_order_preserved=True,
                          standalone_preparation_and_reduction_exact=True,preparation_model_calls=0))
    rejected=0
    for names in ([],['1.dcm','1.dcm'],['bad.dcm'],['.dcm']):
        try:
            select_slice_positions(names)
        except ValueError:
            rejected+=1
        else:
            raise AssertionError('Invalid slice names accepted')
    incomplete=dict(planes)
    incomplete['synthetic-a']=planes['synthetic-a'][1:2]
    try:
        predict(baselines,queries,incomplete,lambda inputs:np.zeros((len(inputs[0]),1),dtype=np.float32))
    except ValueError:
        rejected+=1
    else:
        raise AssertionError('Incomplete source prediction accepted')
    try:
        prepare_slice_inputs(planes['synthetic-a'],47.,'Male','other',decode_selected='invalid')
    except ValueError:
        rejected+=1
    else:
        raise AssertionError('Invalid explicit decoder accepted')
    files=['sciona/osic_image_inference.py','sciona/osic_image_metadata.py',
           'sciona/osic_image_preprocessing.py','sciona/osic_predictions.py',
           'scripts/validate_osic_image_inference.py']
    return dict(format='osic-image-inference.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,source=dict(repository=REPOSITORY,revision=REVISION,normalized_cell_sha256=PINS),
        cases=cases,invalid_contracts_rejected=rejected,
        runtime=dict(numpy=np.__version__,pandas=pd.__version__,opencv=cv2.__version__,opencv_workers=1),
        software_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        limitations=['Synthetic deterministic predictor substitutes for trained B5; trained inference and batch-size-32 parity remain separate qualification.',
            'Synthetic decoded integer planes substitute for file decoding; DICOM parsing is not qualified.',
            'One source image model only; the source multi-model loop repeatedly uses its first result and is not generalized here.',
            'Numerical parity is for the recorded current NumPy runtime, including float32 quantile/scalar rounding; historical binary parity is not claimed.',
            'Int64/float64 baseline coordinates and integer query times only; incomplete slice coverage is rejected.',
            'Publication must bind quantile workflow evidence to the corrected prediction-helper revision.',
            'No original workflow approval or empirical effectiveness claim.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',required=True)
    result=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_image_inference.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(result['cases']),rejected=result['invalid_contracts_rejected'])))
