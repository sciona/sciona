"""Feature-level complete reconstruction with explicit external ensemble sum.

The training operation accepts dense graph features independently of RNA
folding. Its fixed 55 node features, eight adjacency channels, five output
channels and pinned architectures remain recipe constraints. Cross-domain
callers must provide compatible semantics; this is not a universal graph model.
All configured pretraining, supervised steps and refinement rounds are retained.
"""
import copy
import gc
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from sciona.openvaccine_contract import Prepared, _arguments
from sciona.openvaccine_checkpoints import save_checkpoint
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_folding import fold_features
from sciona.openvaccine_models import create_model, validate_inputs
from sciona.openvaccine_population import population_weights, reverse_training_batch
from sciona.openvaccine_round import refine_round
from sciona.openvaccine_splits import validate_member_splits
from sciona.openvaccine_teacher import MemberCheckpoint
from sciona.openvaccine_targets import validation_rollback_required
from sciona.openvaccine_lifecycle_contract import _FIELDS
from sciona.decomposed_openvaccine_ensemble import collect_checkpoints, finish_prediction

_FEATURE_FIELDS = {'nodes', 'adjacency', 'pseudo_nodes', 'pseudo_adjacency',
    'prediction_nodes', 'prediction_adjacency', 'targets', 'cluster_ids',
    'proximity_factors', 'splits', 'pseudo_rounds', 'pretraining_steps',
    'initial_supervised_steps', 'rollback_policy', 'absolute_tolerance', 'std_ddof', 'seed'}


def features(prepared):
    if type(prepared) is not Prepared:
        raise ValueError('Prepared OpenVaccine payload required')
    payload = copy.deepcopy(prepared.payload)
    args = _arguments(payload)
    source = os.environ.get('SCIONA_OPENVACCINE_SOURCE_DIR')
    binary = os.environ.get('SCIONA_OPENVACCINE_FOLD_BINARY')
    parameters = os.environ.get('SCIONA_OPENVACCINE_FOLD_PARAMETERS')
    if not source or not binary or not parameters:
        raise ValueError('Provision reviewed source and folding dependencies')
    n, a = fold_features(source, args.pop('sequences'), binary=binary, parameters=parameters)
    pn, pa = fold_features(source, args.pop('pseudo_sequences'), binary=binary, parameters=parameters)
    qn, qa = fold_features(source, args.pop('prediction_sequences'), binary=binary, parameters=parameters)
    result = dict(args, nodes=n, adjacency=a, pseudo_nodes=pn, pseudo_adjacency=pa,
                  prediction_nodes=qn, prediction_adjacency=qa, seed=payload['seed'])
    checked_features(result)
    return result


def validate_feature_plans(plans, membership, shape, *, rollback_policy, absolute_tolerance, std_ddof):
    if rollback_policy not in ('weights_only', 'model_and_optimizer'):
        raise ValueError('Unsupported rollback policy')
    validation_rollback_required(0., 0., absolute_tolerance=absolute_tolerance)
    if type(std_ddof) is not int or std_ddof not in (0, 1):
        raise ValueError('Explicit integer standard deviation ddof required')
    if not isinstance(plans, (list, tuple)) or not plans:
        raise ValueError('At least one explicit pseudo-label round required')
    for plan in plans:
        if not isinstance(plan,dict) or set(plan)!=EXPECTED_MEMBERS:raise ValueError('All twenty member plans required in every round')
        for key,p in plan.items():
            if not isinstance(p,dict) or set(p)!=_FIELDS:raise ValueError('Exact round-plan fields required')
            for name,size in [('supervised_reverse_flags',len(membership[key].train)),('reverse_flags',shape[0])]:
                flags=np.asarray(p[name])
                if flags.dtype!=np.bool_ or flags.shape!=(size,):raise ValueError('Invalid round reversal decisions')
            eligible=np.asarray(p['eligible']);draws=np.asarray(p['perturbations']);weights=np.asarray(p['sample_weights'])
            if eligible.dtype!=np.bool_ or eligible.shape!=shape:raise ValueError('Invalid round eligibility')
            if draws.dtype!=np.float32 or draws.shape!=shape or not np.isfinite(draws).all():raise ValueError('Invalid round perturbations')
            if (weights.dtype!=np.float32 or weights.shape!=(shape[0],) or not np.isfinite(weights).all()
                    or (weights<0).any() or not np.isfinite(weights.sum()) or weights.sum()<=0):raise ValueError('Invalid round weights')
            if not (eligible & (weights>0)[:,None,None]).any():raise ValueError('No eligible positive-weight pseudo-labels')
            threshold=p['maximum_uncertainty']
            if isinstance(threshold,(bool,str)) or not np.isscalar(threshold) or not np.isfinite(threshold) or threshold<0:
                raise ValueError('Invalid round uncertainty threshold')


def checked_features(featured):
    if type(featured) is not dict or set(featured) != _FEATURE_FIELDS:
        raise ValueError('Exact dense-feature population fields required')
    value = copy.deepcopy(featured)
    if type(value['seed']) is not int or not 0 <= value['seed'] < 2**31:
        raise ValueError('Invalid initialization seed')
    for name in ('pretraining_steps', 'initial_supervised_steps'):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError('Positive explicit training steps required')
    n, a = validate_inputs(value['nodes'], value['adjacency'])
    pn, pa = validate_inputs(value['pseudo_nodes'], value['pseudo_adjacency'])
    qn, qa = validate_inputs(value['prediction_nodes'], value['prediction_adjacency'])
    targets = np.asarray(value['targets'])
    if targets.dtype != np.float32 or targets.shape != (len(n), n.shape[1]-2, 5):
        raise ValueError('Aligned float32 labeled targets required')
    weights = population_weights(value['cluster_ids'], value['proximity_factors'])
    membership = validate_member_splits(targets, weights, value['splits'])
    validate_feature_plans(value['pseudo_rounds'], membership, (len(pn), pn.shape[1]-2, 5),
        rollback_policy=value['rollback_policy'], absolute_tolerance=value['absolute_tolerance'],
        std_ddof=value['std_ddof'])
    return value, (n, a, pn, pa, qn, qa), targets, weights, membership


def fit_forecast(featured):
    value, arrays, targets, weights, membership = checked_features(featured)
    source_dir = os.environ.get('SCIONA_OPENVACCINE_SOURCE_DIR')
    if not source_dir:
        raise ValueError('Provision reviewed model source')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(value['seed'])
    n, a, pn, pa, qn, qa = arrays
    pretraining_steps = value['pretraining_steps']
    initial_supervised_steps = value['initial_supervised_steps']
    pseudo_rounds = value['pseudo_rounds']
    rollback_policy = value['rollback_policy']
    absolute_tolerance = value['absolute_tolerance']
    std_ddof = value['std_ddof']
    batches={};validations={}
    for key,split in membership.items():
        train=np.array(split.train);validation=np.array(split.validation)
        # Fixed reversal decisions are explicit recipe inputs in later rounds.
        batches[key]=(n[train],a[train],targets[train],weights[train])
        validations[key]=(n[validation],a[validation],targets[validation],weights[validation])
    ae=create_model(source_dir,'autoencoder')
    for _ in range(pretraining_steps):ae.train_step(n,a)
    base=ae.base.get_weights();del ae;gc.collect()
    counts=[]
    with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_population_') as temp:
        root=Path(temp);records=[]
        for family,slot in sorted(EXPECTED_MEMBERS):
            runtime=create_model(source_dir,family,base_weights=base)
            for _ in range(initial_supervised_steps):runtime.train_step(*batches[(family,slot)])
            directory=root/'initial'/(family+str(slot));digest=save_checkpoint(runtime,directory)
            records.append(MemberCheckpoint(family,slot,directory,digest))
            del runtime;gc.collect()
        del base
        for index,plan in enumerate(pseudo_rounds):
            recipes={}
            for key in EXPECTED_MEMBERS:
                supplied=plan[key]
                prepared=reverse_training_batch(*batches[key],supplied['supervised_reverse_flags'])
                recipes[key]=dict(supervised_batches=[prepared],validation_batch=validations[key],
                    eligible=supplied['eligible'],maximum_uncertainty=supplied['maximum_uncertainty'],
                    perturbations=supplied['perturbations'],sample_weights=supplied['sample_weights'],
                    reverse_flags=supplied['reverse_flags'])
            records,history=refine_round(source_dir,records,pn,pa,recipes=recipes,
                output_directory=root/('round'+str(index)),std_ddof=std_ddof,
                rollback_policy=rollback_policy,absolute_tolerance=absolute_tolerance)
            counts.append(dict(members=len(records),supervised_steps=sum(h['supervised_steps'] for h in history),
                pseudo_label_steps=sum(h['pseudo_label_section']['steps'] for h in history),
                rollbacks=sum(h['pseudo_label_section']['rolled_back'] for h in history)))
        outputs=collect_checkpoints(source_dir,records,qn,qa,std_ddof=std_ddof)
    return dict(outputs, members=20, rounds=counts,
                pretraining_steps=pretraining_steps, initial_supervised_steps=20*initial_supervised_steps)


def result(forecasted, total):
    prediction = finish_prediction(forecasted, total)
    if set(forecasted) != {'clipped_members', 'teacher_mean', 'teacher_std', 'members',
                          'rounds', 'pretraining_steps', 'initial_supervised_steps'}:
        raise ValueError('Complete same-invocation lifecycle result required')
    if type(forecasted['members']) is not int or forecasted['members'] != 20:
        raise ValueError('Complete twenty-member population required')
    for name in ('pretraining_steps', 'initial_supervised_steps'):
        if type(forecasted[name]) is not int or forecasted[name] < 1:
            raise ValueError('Positive lifecycle counters required')
    if forecasted['initial_supervised_steps'] % 20:
        raise ValueError('Equal configured initial steps per member required')
    rounds = forecasted['rounds']
    if not isinstance(rounds, list) or not rounds:
        raise ValueError('Complete refinement counters required')
    for record in rounds:
        if (type(record) is not dict or set(record) != {'members', 'supervised_steps', 'pseudo_label_steps', 'rollbacks'}
                or any(type(v) is not int for v in record.values())
                or record['members'] != 20 or record['supervised_steps'] != 20
                or record['pseudo_label_steps'] != 20 or not 0 <= record['rollbacks'] <= 20):
            raise ValueError('Complete per-round lifecycle counters required')
    output = dict(predictions=prediction.tolist(), teacher_mean=forecasted['teacher_mean'].tolist(),
                  teacher_std=forecasted['teacher_std'].tolist(), members=20,
                  rounds=copy.deepcopy(rounds), pretraining_steps=forecasted['pretraining_steps'],
                  initial_supervised_steps=forecasted['initial_supervised_steps'])
    json.dumps(output, allow_nan=False)
    return output
