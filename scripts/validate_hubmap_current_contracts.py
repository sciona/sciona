"""Qualify HuBMAP synthetic components; full native and neural execution is separate."""
import json
import importlib.metadata as metadata
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import torch
import pytest
from threadpoolctl import threadpool_limits
from scripts.audit_hubmap_original_readiness import ROOT, sha
from scripts import review_video_numeric_runtime as runtime_checker

TESTS = ['tests/test_hubmap_'+name+'.py' for name in (
    'augmentation','checkpoint','lifecycle','loader','losses','network_contract',
    'preparation','sampling','tiling','validation')]
ROOT_PACKAGES = ['numpy','torch','scipy','scikit-image','opencv-python-headless','threadpoolctl']


def validate():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads()!=1 or cv2.ocl.useOpenCL():
        raise ValueError('Single-threaded OpenCV required')
    if cv2.__version__ != '.'.join(metadata.version('opencv-python-headless').split('.')[:3]):
        raise ValueError('Loaded OpenCV version differs from qualified headless distribution')
    paths=TESTS+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('hubmap_*.py'))]+[
        'scripts/validate_hubmap_current_contracts.py','scripts/review_video_numeric_runtime.py',
        '../sciona-atoms-dl/src/sciona/atoms/dl/segmentation/hubmap_execution.py']
    before={p:sha(ROOT/p) for p in paths}
    previous=runtime_checker.ROOT_PACKAGES
    runtime_checker.ROOT_PACKAGES=ROOT_PACKAGES
    class Collection:
        count=0
        def pytest_collection_finish(self,session):
            self.count=len(session.items)
    collection=Collection()
    try:
        runtime=runtime_checker.review()
        with tempfile.TemporaryDirectory(prefix='synthetic-hubmap-contracts-') as temporary, threadpool_limits(limits=1):
            xml=Path(temporary)/'tests.xml'
            status=pytest.main(['-q',*TESTS,'-W','error::RuntimeWarning','--junitxml='+str(xml)],plugins=[collection])
            counts={k:sum(int(s.get(k,'0')) for s in ET.parse(xml).getroot().iter('testsuite'))
                    for k in ('tests','failures','errors','skipped')}
        if status or collection.count==0 or counts!=dict(tests=collection.count,failures=0,errors=0,skipped=0):
            raise ValueError('Every collected component case must pass without skips')
        if runtime!=runtime_checker.review() or before!={p:sha(ROOT/p) for p in paths}:
            raise ValueError('Software changed during qualification')
    finally:
        runtime_checker.ROOT_PACKAGES=previous
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        tests=counts,test_paths=TESTS,software_sha256=before,runtime=runtime,
        resource_policy=dict(serial_tests=True,torch_threads=1,interop_threads=1,
            native_threads=1,opencv_threads=1,opencv_opencl=False),
        limits=['Component checks only: tiling, preparation, sampling, augmentation, loading, losses, checkpoints, validation and lifecycle routing.',
            'Lifecycle routing tests substitute synthetic training and inference callbacks; they do not qualify full network execution.',
            'Full SE-ResNeXt101 training, pseudo-label generation, fresh retraining, model ensembles and corrected stored graph still require current execution qualification.',
            'Current installed dependency RECORD checks do not authenticate publishers or establish clean-install reproducibility.',
            'OpenCV qualification targets the loaded headless distribution; overlapping older opencv-python metadata has a RECORD mismatch and is not qualified.',
            'Synthetic checks establish implementation behavior, not historical or cross-domain predictive accuracy.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/hubmap_current_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=report['tests'],approved=False)))
