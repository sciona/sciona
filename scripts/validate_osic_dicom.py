"""Verify real encoded synthetic DICOM bytes against pinned source readers."""
import argparse
import ast
import base64
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
WHEEL='pydicom-3.0.1-py3-none-any.whl'
WHEEL_SHA='db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41'
PINS={'training':'603e73e60d97715bf81a1e43c5b77274f48d76c5545c402c73cfa9a4535cb287',
      'inference':'4ff759ede08f197111d0aada45ef7800a0960e6982fb76143b42f377fccdf621'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(source_directory,overlay):
    import cv2
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset,FileMetaDataset
    from pydicom.uid import (ExplicitVRLittleEndian,ImplicitVRLittleEndian,ExplicitVRBigEndian,
        DeflatedExplicitVRLittleEndian,RLELossless,SecondaryCaptureImageStorage,generate_uid)
    from sciona.dicom_planes import decode_plane
    from sciona.osic_dicom import training_image,inference_image
    from scripts.review_osic_original_scope import TRAIN,INFERENCE
    cv2.setNumThreads(0)
    assert cv2.getNumThreads()==1
    for module in (cv2,cv2._native,pydicom):
        assert Path(module.__file__).resolve().is_relative_to(Path(overlay).resolve())
    readers={}
    for branch,name,index in [('training',TRAIN,17),('inference',INFERENCE,9)]:
        notebook=json.loads((Path(source_directory)/name).read_text())
        tree=ast.parse(''.join(notebook['cells'][index]['source']))
        function,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_img']
        assert hashlib.sha256(ast.unparse(function).encode()).hexdigest()==PINS[branch]
        scope=dict(np=np,cv2=cv2,pydicom=pydicom)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<pinned-source-reader>','exec'),scope)
        readers[branch]=scope['get_img']
    syntaxes=[ExplicitVRLittleEndian,ImplicitVRLittleEndian,ExplicitVRBigEndian,
              DeflatedExplicitVRLittleEndian,RLELossless]
    cases=[]
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-dicom-') as tmp:
        for syntax in syntaxes:
            for bits,signed in [(8,False),(8,True),(16,False),(16,True)]:
                values=np.arange(247).reshape(13,19)
                if signed:
                    values=values-123
                values=values.astype(('int' if signed else 'uint')+str(bits))
                dataset=Dataset()
                dataset.file_meta=FileMetaDataset()
                dataset.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian if syntax==RLELossless else syntax
                dataset.SOPClassUID=SecondaryCaptureImageStorage
                dataset.SOPInstanceUID=generate_uid()
                dataset.Rows,dataset.Columns=values.shape
                dataset.SamplesPerPixel=1
                dataset.PhotometricInterpretation='MONOCHROME1' if signed else 'MONOCHROME2'
                dataset.BitsAllocated=bits
                dataset.BitsStored=12 if bits==16 else 8
                dataset.HighBit=dataset.BitsStored-1
                dataset.PixelRepresentation=int(signed)
                dataset.RescaleIntercept='-17.25'
                dataset.RescaleSlope='1.5'
                order='>' if syntax==ExplicitVRBigEndian else '<'
                dataset.PixelData=values.astype(values.dtype.newbyteorder(order)).tobytes()
                if syntax==RLELossless:
                    dataset.compress(RLELossless,encoding_plugin='pydicom')
                path=Path(tmp)/'synthetic.dcm'
                pydicom.dcmwrite(path,dataset,enforce_file_format=True)
                encoded=path.read_bytes()
                decoded=decode_plane(encoded,read_rescale=True)
                np.testing.assert_array_equal(decoded['pixels'],values)
                assert decoded['rescale']==dict(intercept=-17.25,slope=1.5)
                for branch,adapter in [('training',training_image),('inference',inference_image)]:
                    got=adapter(encoded)
                    expected=readers[branch](path)
                    assert got.shape==expected.shape==(512,512)
                    assert got.dtype==expected.dtype
                    np.testing.assert_array_equal(got,expected)
                cases.append(dict(transfer_syntax=syntax.name,bits_allocated=bits,
                    bits_stored=dataset.BitsStored,signed=signed,decoded_values=247,
                    stored_pixels_exact=True,training_and_inference_source_exact=True))
        # Missing rescale metadata is permitted only in the inference branch.
        del dataset.RescaleIntercept
        del dataset.RescaleSlope
        pydicom.dcmwrite(path,dataset,enforce_file_format=True)
        encoded=path.read_bytes()
        np.testing.assert_array_equal(inference_image(encoded),readers['inference'](path))
        rejected=0
        for operation in (lambda:training_image(encoded),lambda:decode_plane(b'invalid',read_rescale=False),
                          lambda:decode_plane(b'',read_rescale=False),lambda:decode_plane(encoded,read_rescale=1)):
            try:
                operation()
            except ValueError:
                rejected+=1
            else:
                raise AssertionError('Invalid decoder contract accepted')
        for geometry in ('color','multiframe'):
            other=copy.deepcopy(dataset)
            other.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
            other.PixelRepresentation=0
            if geometry=='color':
                pixels=np.arange(13*19*3,dtype='uint16').reshape(13,19,3)
                other.SamplesPerPixel=3
                other.PhotometricInterpretation='RGB'
                other.PlanarConfiguration=0
            else:
                pixels=np.arange(2*13*19,dtype='uint16').reshape(2,13,19)
                other.NumberOfFrames=2
            other.PixelData=pixels.astype('<u2').tobytes()
            other['PixelData'].VR='OW'
            other['PixelData'].is_undefined_length=False
            pydicom.dcmwrite(path,other,enforce_file_format=True)
            np.testing.assert_array_equal(pydicom.dcmread(path).pixel_array,pixels)
            try:
                decode_plane(path.read_bytes(),read_rescale=False)
            except ValueError:
                rejected+=1
            else:
                raise AssertionError('Unsupported plane geometry accepted')
    return dict(cases=cases,missing_rescale_inference_passed=True,rejected_contracts=rejected,
        runtime=dict(pydicom=pydicom.__version__,opencv=cv2.__version__,numpy=np.__version__),
        serial_execution=True,imported_decoder_and_opencv_from_overlay=True,
        source_function_sha256=PINS)


def validate(dependencies,source_directory):
    from scripts.audit_osic_opencv_isolation import audit_distribution,digest
    wheel=Path(dependencies)/WHEEL
    if sha(wheel)!=WHEEL_SHA:
        raise ValueError('Pinned decoder wheel changed')
    before=audit_distribution('opencv-python-headless')
    if before['missing'] or before['mismatched']:
        raise ValueError('OpenCV namespace cannot be isolated from changed files')
    selected=metadata.distribution('opencv-python-headless')
    with tempfile.TemporaryDirectory(prefix='sciona-osic-dicom-overlay-') as temporary:
        overlay=Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(overlay)
            license_digest=hashlib.sha256(archive.read('pydicom-3.0.1.dist-info/LICENSE')).hexdigest()
        copied={}
        for entry in selected.files or []:
            relative=Path(str(entry))
            if not entry.hash or relative.parts[0]!='cv2':
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe installed package path')
            target=overlay/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(selected.locate_file(entry),target)
            assert base64.urlsafe_b64encode(digest(target,entry.hash.mode)).rstrip(b'=').decode()==entry.hash.value
            copied[str(relative)]=sha(target)
        env=dict(os.environ,PYTHONPATH=str(overlay)+os.pathsep+str(ROOT),PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[key]='1'
        process=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
            '--source-directory',str(source_directory),'--dependencies',str(overlay)],
            env=env,capture_output=True,text=True)
        if process.returncode:
            raise RuntimeError('Synthetic DICOM qualification failed:\n'+process.stderr)
        result=json.loads(process.stdout)
        assert copied=={name:sha(overlay/name) for name in copied}
    assert audit_distribution('opencv-python-headless')==before
    files=['sciona/dicom_planes.py','sciona/osic_dicom.py','sciona/osic_image_preprocessing.py',
           'scripts/validate_osic_dicom.py']
    return dict(format='osic-dicom.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,shared_environment_mutations=0,execution=result,
        decoder_wheel_sha256=WHEEL_SHA,decoder_license_sha256=license_digest,
        opencv_isolated_files=len(copied),
        software_sha256={name:sha(ROOT/name) for name in files},
        limitations=['Twenty synthetic monochrome cases across five transfer syntaxes; JPEG-family codecs are not qualified.',
            'Single-frame integer planes only; color, multiframe, float-pixel and presentation transformations are outside this adapter contract.',
            'Synthetic pydicom writer round trips plus independent integer-array equality and pinned source preprocessing comparisons; not a DICOM conformance certification.',
            'Historical decoder/runtime equivalence and real-data effectiveness are not claimed.',
            'No original workflow, dependency redistribution or catalog approval.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies',required=True)
    parser.add_argument('--source-directory',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.source_directory,args.dependencies)))
    else:
        report=validate(args.dependencies,args.source_directory)
        (ROOT/'docs/reviews/osic_dicom.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=True,cases=len(report['execution']['cases']),shared_environment_mutations=0)))
