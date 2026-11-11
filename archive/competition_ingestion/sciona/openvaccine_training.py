"""Explicit reconstructed member lifecycle; training recipe supplied by caller.

Pretraining batches contain features/adjacency. Supervised and pseudo-label
sections contain fully prepared feature/adjacency/target/weight batches.
Population/fold provenance, draws and teacher versions belong to the outer
lifecycle and must not be inferred from these tensors.
"""
import gc

import numpy as np

from sciona.openvaccine_checkpoints import save_checkpoint
from sciona.openvaccine_models import create_model, validate_inputs
from sciona.openvaccine_sections import run_pseudo_label_section
from sciona.openvaccine_targets import validation_rollback_required


def train_member(source_dir, family, *, pretraining_batches, supervised_sections,
                 pseudo_label_sections, validation_batch, checkpoint_directory,
                 rollback_policy, absolute_tolerance):
    if family not in ('gru','lstm','forward','wave'):
        raise ValueError('Prediction family required')
    if rollback_policy not in ('weights_only','model_and_optimizer'):
        raise ValueError('Explicit rollback policy required')
    validation_rollback_required(0.,0.,absolute_tolerance=absolute_tolerance)
    if not pretraining_batches or not supervised_sections or len(supervised_sections)!=len(pseudo_label_sections):
        raise ValueError('Pretraining and paired supervised/pseudo-label sections required')
    if any(not section for section in list(supervised_sections)+list(pseudo_label_sections)):
        raise ValueError('Empty training sections are unsupported')
    vn,va,vt,vw=validation_batch
    validate_inputs(vn,va)
    vt,vw=np.asarray(vt),np.asarray(vw)
    if (vt.dtype!=np.float32 or vt.shape!=(len(vn),vn.shape[1]-2,5)
            or np.isinf(vt).any() or vw.dtype!=np.float32
            or vw.shape!=(len(vn),) or not np.isfinite(vw).all()
            or (vw<0).any() or not np.isfinite(vw.sum()) or vw.sum()<=0):
        raise ValueError('Validation requires valid masked targets and weights')
    if not (np.isfinite(vt) & (vw>0)[:,None,None]).any():
        raise ValueError('Validation needs positive-weight observed targets')
    autoencoder=create_model(source_dir,'autoencoder')
    ae_losses=[autoencoder.train_step(*batch) for batch in pretraining_batches]
    learned=autoencoder.base.get_weights()
    del autoencoder
    gc.collect()
    runtime=create_model(source_dir,family,base_weights=learned)
    del learned
    def score(model):
        tf=model.tf
        return float(model.model.loss(tf.constant(vt),tf.constant(model.predict(vn,va)),tf.constant(vw)).numpy())
    history=[]
    for supervised,pseudo in zip(supervised_sections,pseudo_label_sections):
        losses=[runtime.train_step(*batch) for batch in supervised]
        history.append(dict(kind='supervised',steps=len(losses),mean_training_loss=float(np.mean(losses)),validation=score(runtime)))
        result=run_pseudo_label_section(runtime,pseudo,score,absolute_tolerance=absolute_tolerance,rollback_policy=rollback_policy)
        history.append(dict(kind='pseudo_label',**result))
    final_score=score(runtime)
    if not np.isfinite(final_score):
        raise ValueError('Nonfinite final validation')
    digest=save_checkpoint(runtime,checkpoint_directory)
    return runtime,dict(family=family,pretraining_steps=len(ae_losses),sections=history,
                        final_validation=final_score,checkpoint_manifest_sha256=digest,
                        validation_metric='Source weighted per-example per-target RMSE with epsilon; NaN masks retain full-position denominator',
                        stochastic_replay_claim=False)
