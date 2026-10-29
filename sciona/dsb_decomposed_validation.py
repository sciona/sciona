"""Aggregate-only source validation metrics for runtime-provided batches."""
import numpy as np
import torch
from sciona.dsb_losses import detector_loss
from sciona.dsb_decomposed_losses import classifier_loss


def validate_detector(model,batches):
    """Source averages batch losses equally and sums classification counts."""
    model.eval();parameter=next(model.parameters());rows=[]
    with torch.inference_mode():
        for images,labels,coords in batches:
            images,labels,coords=[torch.as_tensor(v,device=parameter.device,dtype=parameter.dtype) for v in (images,labels,coords)]
            _,predicted=model(images,coords)
            loss=detector_loss(predicted,labels,training=False)
            rows.append([float(loss['total']),float(loss['classification']),*loss['regression'].tolist(),
                         loss['positive_correct'],loss['positive_count'],loss['negative_correct'],loss['negative_count']])
    if not rows:raise ValueError('validation requires at least one batch')
    metrics=np.asarray(rows,dtype=np.float32)
    totals=metrics[:,6:].sum(axis=0)
    return dict(batch_count=len(rows),mean_losses=metrics[:,:6].mean(axis=0).tolist(),
                positive_correct=int(totals[0]),positive_count=int(totals[1]),
                negative_correct=int(totals[2]),negative_count=int(totals[3]),
                true_positive_rate=float(totals[0]/totals[1]) if totals[1] else None,
                true_negative_rate=float(totals[2]/totals[3]) if totals[3] else None)


def validate_classifier(model,batches):
    """Source weights BCE, miss loss, and accuracy by case count, strict p>0.5."""
    model.eval();parameter=next(model.parameters());count=0;losses=[];sizes=[];accuracies=[]
    tp=fp=fn=0
    with torch.inference_mode():
        for images,coords,known,labels in batches:
            images,coords,known,labels=[torch.as_tensor(v,device=parameter.device,dtype=parameter.dtype) for v in (images,coords,known,labels)]
            labels=labels.reshape(-1)
            _,case,each=model(images,coords)
            loss=classifier_loss(case,each,labels,known)
            truth=labels.cpu().numpy();predicted=case.cpu().numpy()>.5
            tp+=int(np.sum(predicted[truth==1]));fp+=int(np.sum(predicted[truth==0]));fn+=int(np.sum(~predicted[truth==1]))
            size=len(images);count+=size;sizes.append(size)
            losses.append([float(loss['classification']),float(loss['miss'])]);accuracies.append(float(np.mean(truth==predicted)))
    if not count:raise ValueError('validation requires at least one case')
    means=(np.asarray(losses)*np.asarray(sizes)[:,None]).sum(axis=0)/count
    return dict(case_count=count,classification_loss=float(means[0]),miss_loss=float(means[1]),
                accuracy=float(np.sum(np.array(accuracies)*sizes)/count),true_positives=tp,false_positives=fp,false_negatives=fn)
