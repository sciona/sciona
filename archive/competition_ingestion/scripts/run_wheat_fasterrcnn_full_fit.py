"""Run the complete published Faster R-CNN fit on generated box images."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from sciona.wheat_augmentation_compat import historical_augmentation_api
from sciona.wheat_base_population import base_populations
from sciona.wheat_dataset import WheatDataset
from sciona.wheat_fasterrcnn_fit import fit_fasterrcnn


class SyntheticBoxLoader:
    def __call__(self,key):
        key=int(key)
        rng=np.random.default_rng(1597+key)
        width=1408 if key>=120 else 1024
        image=rng.integers(15,35,(1024,width,3),dtype=np.uint8)
        boxes=[]
        for offset in (0,1):
            x=int(rng.integers(60+offset*430,150+offset*430))
            y=int(rng.integers(100+offset*310,190+offset*310))
            right=min(x+int(rng.integers(220,330)),width)
            bottom=min(y+int(rng.integers(200,290)),1024)
            image[y:bottom,x:right]=rng.integers(185,235,(bottom-y,right-x,3),dtype=np.uint8)
            boxes.append([x,y,right,bottom])
        return image,np.asarray(boxes,dtype=float),'spike' if key>=120 else 'primary'


def main(runtime,checkpoint,augmentation_root,output):
    root=Path(__file__).resolve().parents[1]
    runtime=runtime.resolve()
    if runtime==root or root in runtime.parents:raise ValueError('private runtime outside repository required')
    runtime.mkdir(parents=True,exist_ok=False)
    report_names=['workers','dataset','base_schedule','base_control','validation_metric','mixup','assembled_reference']
    evidence={}
    for name in report_names:
        path=root/f'docs/reviews/competition_global_wheat_{name}.json'
        raw=path.read_bytes();report=json.loads(raw)
        if not report.get('passed'):raise ValueError('missing prerequisite qualification')
        for file,digest in report.get('implementation_sha256',report.get('post_execution_implementation_consistency_sha256',{})).items():
            if hashlib.sha256((root/file).read_bytes()).hexdigest()!=digest:raise ValueError('qualified implementation drift: '+file)
        evidence[name]=hashlib.sha256(raw).hexdigest()
    dependencies={}
    for package in ('albumentations','imgaug'):
        manifest=json.loads((augmentation_root/(package+'_manifest.json')).read_text())
        for name,digest in manifest['files_sha256'].items():
            if hashlib.sha256((augmentation_root/package/name).read_bytes()).hexdigest()!=digest:raise ValueError('augmentation dependency drift')
        dependencies[package]=manifest
    files=list((root/'sciona').glob('wheat_*.py'))+[Path(__file__)]
    source_digests={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    random.seed(1597);np.random.seed(1597);torch.manual_seed(1597);torch.set_num_threads(2)
    with historical_augmentation_api():
        import imgaug
        import albumentations as A
        import cv2
        if not Path(A.__file__).resolve().is_relative_to((augmentation_root/'albumentations').resolve()):raise ValueError('incorrect augmentation import')
        imgaug.seed(1597);cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
        primary=pd.DataFrame(dict(image_id=np.arange(100),fold=np.tile(np.arange(5),20),isbox=True))
        auxiliary=pd.DataFrame(dict(image_id=np.arange(100,120),fold=-1,isbox=True))
        warm_only=pd.DataFrame(dict(image_id=np.arange(120,140),fold=-1,isbox=True))
        populations=base_populations(primary,auxiliary,warm_only,1,numpy_rng=np.random.mtrand._rand)
        loader=SyntheticBoxLoader()
        warm=WheatDataset(populations['warm']['image_id'].to_numpy(),loader,1024)
        training=WheatDataset(populations['training']['image_id'].to_numpy(),loader,1024)
        validation=WheatDataset(populations['validation']['image_id'].to_numpy(),loader,1024,mode='valid')
        context=dict(seed=1597,source_sha256=source_digests,prerequisite_report_sha256=evidence,
            dependencies=dependencies,synthetic_only=True,population_counts=dict(warm=len(warm),training=len(training),validation=len(validation)),
            controls=dict(batch_size=20,workers=16,warm_epochs=20,epoch_ceiling=100,patience=40),
            versions=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,albumentations=A.__version__,imgaug=imgaug.__version__))
        (runtime/'context.json').write_text(json.dumps(context,indent=2)+'\n')
        print(json.dumps(dict(started=True,**context['population_counts'],**context['controls'])),flush=True)
        result=fit_fasterrcnn(warm,training,validation,checkpoint,runtime/'fit',record=lambda row:print(json.dumps(row),flush=True))
    for name,digest in source_digests.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('fit implementation changed during execution')
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,**result,
        source_sha256=source_digests,prerequisite_report_sha256=evidence,
        limits=['One complete synthetic Faster R-CNN base fit; remaining nine base fits, pseudo-fits and ensemble remain pending.',
                'Checkpoint replay and publication qualification remain separate.',
                'CPU execution with qualified historical Python semantics on installed native kernels.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','prerequisite_report_sha256')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-directory',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--augmentation-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.runtime_directory,args.checkpoint,args.augmentation_root,args.output)
