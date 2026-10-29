"""Source epoch lifecycle with reusable classifier penalty in training/validation.

Preserves all source passes and per-epoch checkpoint emission.
"""
from sciona.dsb_components import _integer
from sciona.dsb_schedule import epoch_steps


def run_training_epoch(detector, classifier, detector_optimizer, classifier_optimizer,
                       batch_factory, *, epoch, start_epoch, classifier_variant=3,
                       freeze_batchnorm=False, learning_rate_override=None, after_pass=None):
    """Execute source-ordered training passes using caller-supplied batch factories.

    batch_factory(task, augmentation_profile) is called anew for each pass, so
    the initial debug pass cannot exhaust a one-shot loader for the later pass.
    Detector batches are (images, labels, coordinates); classifier batches are
    (images, coordinates, known_proposals, case_labels). Validation/checkpoint
    saving remain separate operations; returned evidence is aggregate only.
    """
    import torch
    from sciona.dsb_schedule import CLASSIFIER_PROFILES, epoch_steps, training_mode
    from sciona.dsb_decomposed_losses import detector_loss
    from sciona.dsb_decomposed_losses import classifier_loss
    if classifier.NoduleNet is not detector:
        raise ValueError('alternating training requires the same feature network instance')
    for model, optimizer in [(detector,detector_optimizer),(classifier,classifier_optimizer)]:
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        if len(parameters)!=len({id(p) for p in parameters}) or {id(p) for p in parameters}!={id(p) for p in model.parameters()}:
            raise ValueError('optimizer parameters do not match the source task network')
    report=[]
    for step in epoch_steps(epoch,start_epoch,classifier_variant,learning_rate_override):
        task=step['task']
        model,optimizer=(detector,detector_optimizer) if task=='detector' else (classifier,classifier_optimizer)
        profile=(dict(flip=True,swap=False,scale=True,rotate=False) if task=='detector'
                 else {key:CLASSIFIER_PROFILES[classifier_variant][key] for key in ['flip','swap','scale','rotate']})
        training_mode(model,freeze_batchnorm)
        for group in optimizer.param_groups:group['lr']=step['learning_rate']
        parameter=next(model.parameters())
        total,examples,batches=0.,0,0
        for index,batch in enumerate(batch_factory(task,profile)):
            if step['max_batches'] is not None and index>=step['max_batches']:break
            values=[torch.as_tensor(v,device=parameter.device,dtype=parameter.dtype) for v in batch]
            # Source-era PyTorch zero_grad cleared existing buffers rather than
            # replacing them with None; this matters for shared unused heads.
            optimizer.zero_grad(set_to_none=False)
            if task=='detector':
                images,labels,coordinates=values
                _,prediction=model(images,coordinates)
                loss=detector_loss(prediction,labels,num_hard=2,training=True)['total']
            else:
                images,coordinates,known,labels=values
                _,case,each=model(images,coordinates)
                loss=classifier_loss(case,each,labels.reshape(-1),known)['total']
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite training loss')
            loss.backward();optimizer.step()
            count=images.shape[0]
            total+=float(loss.detach())*count;examples+=count;batches+=1
        report.append(dict(task=task,learning_rate=step['learning_rate'],batches=batches,
                           examples=examples,mean_loss=total/examples if examples else None))
        if after_pass is not None and step['max_batches'] is None:
            after_pass(task,model)
    return report


def run_epoch_lifecycle(detector,classifier,detector_optimizer,classifier_optimizer,
                        training_factory,validation_factory,*,epoch,start_epoch,
                        save_frequency=1,**kwargs):
    """Source-ordered training, validation and optional in-memory checkpoint.

    Validation factory keys distinguish detector, classifier held-out, and
    classifier training-set evaluation. Caller owns private array loading and
    checkpoint storage; no filenames or data provenance enter returned metrics.
    """
    from sciona.dsb_decomposed_validation import validate_detector,validate_classifier
    from sciona.dsb_checkpoint import checkpoint_bytes
    save_frequency=_integer(save_frequency,'save_frequency')
    validation=[]
    def after_pass(task,model):
        if task=='detector':
            validation.append(dict(task=task,metrics=validate_detector(model,validation_factory('detector'))))
        else:
            for key in ['classifier_validation','classifier_training_validation']:
                validation.append(dict(task=key,metrics=validate_classifier(model,validation_factory(key))))
    training=run_training_epoch(detector,classifier,detector_optimizer,classifier_optimizer,
        training_factory,epoch=epoch,start_epoch=start_epoch,after_pass=after_pass,**kwargs)
    checkpoint=checkpoint_bytes(classifier,epoch) if epoch%save_frequency==0 else None
    return dict(training=training,validation=validation,checkpoint=checkpoint)


def run_training_range(runtime,training_factory,validation_factory,*,end_epoch,
                       classifier_variant=3,freeze_batchnorm=False,prepare_epoch=None):
    """Keep momentum, shared model buffers and RNG streams across the epoch range.

    runtime comes from one restore_training_runtime call. prepare_epoch, when
    supplied, receives each epoch before its passes and can install caller-owned
    sample orders. The factory is never recreated or reseeded here. Returned
    checkpoint bytes contain only the final model; metrics contain no model states.
    """
    start=_integer(runtime['start_epoch'],'start_epoch')
    end=_integer(end_epoch,'end_epoch')
    if end<start:raise ValueError('end_epoch precedes restored start_epoch')
    # Reject unsupported ranges before the first optimizer update.
    for epoch in range(start,end+1):epoch_steps(epoch,start,classifier_variant)
    reports=[]
    for epoch in range(start,end+1):
        if prepare_epoch is not None:prepare_epoch(epoch)
        result=run_epoch_lifecycle(**runtime,training_factory=training_factory,
            validation_factory=validation_factory,epoch=epoch,classifier_variant=classifier_variant,
            freeze_batchnorm=freeze_batchnorm,save_frequency=1)
        reports.append(dict(epoch=epoch,training=result['training'],validation=result['validation']))
    return dict(checkpoint=result['checkpoint'],epochs=reports)
