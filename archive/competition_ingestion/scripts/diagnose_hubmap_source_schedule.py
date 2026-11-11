"""Diagnose synthetic pseudo-mask failure with one real fold over the source schedule; never qualifies a CDG."""
import gc
import json
import random
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.audit_hubmap_original_readiness import ROOT,sha
from scripts.validate_hubmap_current_graph import prerequisites
from scripts.hubmap_synthetic_rasters import labeled_raster
from sciona.hubmap_encoder import build_encoder
from sciona.hubmap_lifecycle import FoldStage,train_fold
from sciona.hubmap_raw_inference import load_inference_model,predict_slide


def main():
    prerequisites()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
    torch.manual_seed(367)
    encoder=build_encoder();state=encoder.state_dict();del encoder
    slides=[];groups=[]
    for fold in range(4):
        for foreground in (False,True):
            slides.append(labeled_raster(410+fold*2+int(foreground),foreground))
            groups.append('synthetic-group-'+str(fold))
    stage=FoldStage(state,0,['synthetic-group-0'],dict(start_epoch=1,end_epoch=20,patience=50,maximum_bin=4,
        training_batch_size=2,validation_batch_size=2,input_side=320,python_rng=random.Random(0),
        numpy_rng=np.random.RandomState(1),torch_generator=torch.Generator().manual_seed(2)))
    observations=[]
    with threadpool_limits(limits=1):
        trained=train_fold(stage,slides,groups,tile_size=1024)
        model=load_inference_model(trained['checkpoints']['best_score'],input_resolution=320)
        counters=dict(classifier_calls=0,decoder_calls=0,classifier_positive_examples=0)
        classifier_range=[];segmentation_range=[]
        def classifier(module,args,output):
            counters['classifier_calls']+=1
            p=output.detach().sigmoid()
            counters['classifier_positive_examples']+=int((p>.5).sum())
            classifier_range.extend([float(p.min()),float(p.max())])
        def decoder(module,args,output):counters['decoder_calls']+=1
        def segmentation(module,args,output):
            p=output.detach().sigmoid();segmentation_range.extend([float(p.min()),float(p.max())])
        hooks=[model.clf.register_forward_hook(classifier),model.center.register_forward_hook(decoder),
               model.final_conv.register_forward_hook(segmentation)]
        try:
            image,_=labeled_raster(479,True)
            mask=predict_slide([model],image,python_rng=random.Random(489),
                torch_generator=torch.Generator().manual_seed(497),resolution=1024,input_resolution=320,
                pad_size=256,batch_size=2,tta=4,threshold=.5)
        finally:
            for hook in hooks:hook.remove()
        observations=dict(**counters,foreground_pixels=int(mask.sum()),
            classifier_probability_range=[min(classifier_range),max(classifier_range)],
            segmentation_probability_range=[min(segmentation_range),max(segmentation_range)] if segmentation_range else None)
        epochs=[dict(epoch=e['epoch'],training_loss=e['training']['loss'],training_dice=e['training']['dice'],
            validation_loss=e['validation']['loss'],validation_dice=e['validation']['dice'],saved=e['decision']['save'])
            for e in trained['epochs']]
    report=dict(format='hubmap-source-schedule-diagnostic.v1',approved=False,catalog_mutations=0,synthetic_only=True,
        folds_executed=1,configured_epochs=20,patience=50,selected_checkpoint="best_score",epochs=epochs,observations=observations,
        resource_policy=dict(torch_threads=1,interop_threads=1,native_threads=1,opencv_threads=1,data_loader_workers=0),
        diagnostic_sha256=sha(__file__),fixture_sha256=sha(ROOT/'scripts/hubmap_synthetic_rasters.py'),
        limits=['One-fold failure diagnosis only; this is not complete CDG execution evidence or predictive-quality evaluation.',
                'Hooks observe outputs without modifying logits, thresholds, checkpoints or model execution.'])
    (ROOT/'docs/reviews/hubmap_source_schedule_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
