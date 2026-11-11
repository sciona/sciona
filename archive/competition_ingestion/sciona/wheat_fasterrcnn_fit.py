"""Complete source-budget Faster R-CNN fit with durable epoch evidence."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from sciona.wheat_base_control import BaseTrainingControl
from sciona.wheat_base_schedule import base_learning_rate
from sciona.wheat_fasterrcnn import initialized_fasterrcnn
from sciona.wheat_mixup import mixup_batch
from sciona.wheat_sampler import WheatRandomSampler
from sciona.wheat_validation_metric import fasterrcnn_validation_score
from sciona.wheat_workers import WheatWorkerDataset, collate_wheat, source_loader_iterator


def _write_json(path,value):
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def fit_fasterrcnn(warm,training,validation,checkpoint,runtime,*,record):
    """Run 100 epochs or source patience 40, batch 20, with 16 fresh workers.

    Runtime is private and fresh. Checkpoint-free zero-AP fits fail explicitly.
    Caller binds code, dependency versions and synthetic population provenance.
    """
    runtime=Path(runtime)
    runtime.mkdir(parents=True,exist_ok=False)
    if min(len(warm),len(training))<20 or len(validation)<1:
        raise ValueError('complete source training batches and validation population required')
    def loader(dataset,train):
        return DataLoader(WheatWorkerDataset(dataset),batch_size=20,
            sampler=WheatRandomSampler(dataset) if train else SequentialSampler(dataset),
            num_workers=16,drop_last=train,pin_memory=False,collate_fn=collate_wheat,
            multiprocessing_context='spawn',persistent_workers=False,prefetch_factor=2)
    warm_loader,train_loader,valid_loader=loader(warm,True),loader(training,True),loader(validation,False)
    model=initialized_fasterrcnn(checkpoint)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(parameters,lr=.0005,momentum=.9,weight_decay=.0005)
    control=BaseTrainingControl('fasterrcnn')
    history=[];total_updates=0;best_digest=None
    for epoch in range(100):
        rate=base_learning_rate(epoch,'fasterrcnn')
        optimizer.param_groups[0]['lr']=rate
        model.train();total_loss=0.;examples=0;updates=0;skipped=0
        current=warm_loader if epoch<20 else train_loader
        for images,targets in source_loader_iterator(current):
            images,targets=mixup_batch(images,targets,epoch,'fasterrcnn',python_rng=random,
                                       numpy_rng=np.random,torch_generator=None)
            targets=[dict(target) for target in targets]
            optimizer.zero_grad(set_to_none=False)
            losses=sum(model(list(images),targets).values())
            if losses==0 or not torch.isfinite(losses):
                skipped+=1
                continue
            losses.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError('nonfinite full-fit gradient')
            optimizer.step()
            total_loss+=float(losses.detach())*len(images);examples+=len(images);updates+=1
        model.eval();predictions=[]
        with torch.no_grad():
            for images,targets in source_loader_iterator(valid_loader):
                outputs=model(list(images))
                for target,prediction in zip(targets,outputs):
                    predictions.append(dict(pred_boxes=prediction['boxes'].cpu().numpy(),
                        scores=prediction['scores'].cpu().numpy(),gt_boxes=target['boxes'].cpu().numpy()))
        score=fasterrcnn_validation_score(predictions,1024)
        decision=control.observe(epoch,score)
        if decision['save_checkpoint']:
            temporary=runtime/'best.pending'
            torch.save(model.state_dict(),temporary)
            temporary.replace(runtime/'best.pt')
            best_digest=hashlib.sha256((runtime/'best.pt').read_bytes()).hexdigest()
        total_updates+=updates
        row=dict(epoch=epoch,learning_rate=rate,population='warm' if epoch<20 else 'training',
            training_loss=total_loss/examples if examples else 0.,validation_score=score,
            updates=updates,skipped_batches=skipped,**decision)
        history.append(row)
        _write_json(runtime/'history.json',history)
        _write_json(runtime/'progress.json',dict(completed_epochs=len(history),optimizer_updates=total_updates,
            best_checkpoint_sha256=best_digest,best_epoch=control.best_epoch,source_stopping_reached=control.stopped))
        record(row)
        if decision['stop']:break
    selected=control.selected_epoch()
    if best_digest is None or total_updates==0:
        raise ValueError('full fit lacks a trained source-selected checkpoint')
    result=dict(completed_epochs=len(history),optimizer_updates=total_updates,best_epoch=selected,
        best_validation_score=control.best,best_checkpoint_sha256=best_digest,source_stopping_reached=True)
    _write_json(runtime/'result.json',result)
    return result
