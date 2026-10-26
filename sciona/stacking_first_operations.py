"""Draft first-level operations preserving all source entries and shared KNN fits.

Computational entry selection is explicit; the source policy fixes nine classes,
five folds, at least thirteen features and the declared transductive embedding.
These adapters are not the complete source graph and have no catalog approval.
"""
from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
import numpy as np
from sciona.otto_preprocessing import representation
from sciona.otto_meta_layout import AlignedBlock
from sciona.otto_tsne import PopulationEmbedding
from sciona.otto_tsne_clusters import fit_embedding_clusters


@dataclass(frozen=True, repr=False)
class FirstInputs:
    args: tuple
    embedding: PopulationEmbedding
    seed: int
    controls: dict


@dataclass(frozen=True, repr=False)
class NeighborBank:
    inputs: FirstInputs
    controls_key: str
    raw: dict


def prepare_first_inputs(training, labels, folds, training_ids, query, query_ids,
                         embedding, seed, controls):
    x=representation(training,kind='raw');q=representation(query,kind='raw');y=np.asarray(labels);f=np.asarray(folds)
    if x.shape[1]<13 or q.shape[1]!=x.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):raise ValueError('Aligned nine-class populations with at least thirteen features required')
    if f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):raise ValueError('Five aligned folds required')
    seen=set()
    for ids,size in ((training_ids,len(x)),(query_ids,len(q))):
        if type(ids) is not list or len(ids)!=size or any(type(i) is not str or not i for i in ids):raise ValueError('Aligned opaque identities required')
        for identity in ids:
            if identity in seen:raise ValueError('Duplicate or overlapping identities')
            seen.add(identity)
    for fold in range(5):
        if np.sum(f!=fold)<1024 or any(np.sum((f!=fold)&(y==label))<4 for label in range(9)):raise ValueError('Every fold needs 1024 references and four of each class')
    if not isinstance(embedding,PopulationEmbedding):raise ValueError('Fixed embedding required')
    embedding.lookup(training_ids);embedding.lookup(query_ids)
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Common signed integer seed required')
    if type(controls) is not dict or any(type(i) is not int for i in controls) or set(controls)!=set(range(1,34)) or any(type(c) is not dict for c in controls.values()):raise ValueError('Explicit controls for all 33 entries required')
    y=y.copy();f=f.copy()
    for array in (x,q,y,f):array.setflags(write=False)
    coordinates=embedding.coordinates.copy();coordinates.setflags(write=False)
    detached=PopulationEmbedding(tuple(embedding.identities),coordinates,embedding.kl_divergence)
    return FirstInputs((x,y,f,list(training_ids),q,list(query_ids)),detached,seed,deepcopy(controls))


def _context(inputs):
    if type(inputs) is not FirstInputs:raise ValueError('Prepared first-level inputs required')
    return inputs.args


def _block(inputs, entry, a, b):
    x,y,f,training_ids,q,query_ids=_context(inputs)
    a=np.asarray(a);b=np.asarray(b);width=5 if entry==10 else 9
    if a.shape!=(len(x),width) or b.shape!=(len(q),width) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Producer emitted invalid first-level block')
    return AlignedBlock(a,b,tuple(training_ids),tuple(query_ids),None if entry==10 else f.copy())


def fit_entry(inputs, entry):
    """Execute exactly one of source entries 1 through 23, with its own controls."""
    args=_context(inputs)
    if type(entry) is not int or not 1<=entry<=23:raise ValueError('Explicit source entry from 1 through 23 required')
    options=inputs.controls[entry];embedding=inputs.embedding;seed=inputs.seed
    training_ids=args[3];query_ids=args[5]
    def call(name,function,**kwargs):
        return getattr(importlib.import_module('sciona.otto_'+name),function)(*args,**kwargs)
    if entry==10:
        model=fit_embedding_clusters(embedding,training_ids,seed=seed,**options)
        return _block(inputs,entry,model.lookup(training_ids),model.lookup(query_ids))
    if entry==1:r=call('random_forest_crossfit','crossfit_forest',seed=seed,**options)
    elif entry in (2,3,7):r=call('classical_models','crossfit',family={2:'logistic',3:'extra_trees',7:'multinomial_nb'}[entry],seed=seed,controls=options)
    elif entry in (4,22,23):r=call('variant_knn','crossfit',variant={4:'scaled_log',22:'raw_zero',23:'raw_zero_log'}[entry],**options)
    elif entry==5:r=call('libfm_crossfit','crossfit_libfm',seed=seed,**options)
    elif entry==6:r=call('h2o_crossfit','crossfit_h2o',seed=seed,controls=options)
    elif entry in (8,9):r=call('lasagne_crossfit','crossfit_lasagne',variant='pair' if entry==8 else 'six_log',seed=seed,controls=options)
    elif entry==11:r=call('sofia_crossfit','crossfit_sofia',seed=seed,**options)
    elif entry in (12,13):r=call('sofia_interactions_crossfit' if entry==12 else 'sofia_roc_crossfit','crossfit_sofia',embedding=embedding,seed=seed,controls=options)
    elif entry in (14,15,21):r=call({14:'ovr_crossfit',15:'augmented_crossfit',21:'raw_boosting_crossfit'}[entry],'crossfit_boosting',seed=seed,controls=options)
    elif entry in (16,17,18):r=call('embedding_crossfit','crossfit_boosting',embedding=embedding,variant={16:'raw',17:'log1p',18:'scaled_raw'}[entry],seed=seed,controls=options)
    elif entry in (19,20):r=call('lasagne_120_crossfit','crossfit_lasagne',variant='two_hidden' if entry==19 else 'three_hidden',seed=seed,controls=options)
    return _block(inputs,entry,r['oof'],r['query'])


def neighbor_groups(controls):
    """Partition entries 24..33 by exactly the historical JSON control key."""
    groups={}
    for entry in range(24,34):
        key=json.dumps(controls[entry],sort_keys=True)
        groups.setdefault(key,[]).append(entry)
    return list(groups.values())


def fit_neighbor_bank(inputs, representative):
    """Fit one shared ten-column bank for the representative's explicit controls."""
    args=_context(inputs)
    if type(representative) is not int or not 24<=representative<=33:
        raise ValueError('Explicit neighbor entry from 24 through 33 required')
    options=inputs.controls[representative]
    from sciona.otto_knn_models import crossfit
    raw=crossfit(*args,**options)
    for key,size in (('oof',len(args[0])),('query',len(args[4]))):
        array=np.asarray(raw[key])
        if array.shape!=(size,10,9) or not np.isfinite(array).all():
            raise ValueError('Complete finite ten-column neighbor bank required')
    return NeighborBank(inputs,json.dumps(options,sort_keys=True),raw)


def select_neighbor_entry(inputs, bank, entry):
    """Select one neighbor count without refitting or mixing control groups."""
    _context(inputs)
    if type(entry) is not int or not 24<=entry<=33:raise ValueError('Explicit neighbor entry required')
    if type(bank) is not NeighborBank or bank.inputs is not inputs:
        raise ValueError('Neighbor bank from these prepared inputs required')
    if bank.controls_key!=json.dumps(inputs.controls[entry],sort_keys=True):
        raise ValueError('Neighbor control group differs')
    column=entry-24
    return _block(inputs,entry,bank.raw['oof'][:,column,:],bank.raw['query'][:,column,:])


def build_decomposed(training,labels,folds,training_ids,query,query_ids,*,embedding,seed,controls,progress=None):
    """Sequential reference composition; the CDG must expose each operation."""
    inputs=prepare_first_inputs(training,labels,folds,training_ids,query,query_ids,embedding,seed,controls)
    blocks={}
    for entry in range(1,24):
        blocks[entry]=fit_entry(inputs,entry)
        if progress is not None:progress(entry)
    banks={}
    for entry in range(24,34):
        key=json.dumps(inputs.controls[entry],sort_keys=True)
        if key not in banks:banks[key]=fit_neighbor_bank(inputs,entry)
        blocks[entry]=select_neighbor_entry(inputs,banks[key],entry)
        if progress is not None:progress(entry)
    return blocks


def collect_first_blocks(blocks):
    """Assemble all 33 source entries with explicit row/fold alignment checks."""
    if type(blocks) is not dict or any(type(k) is not int for k in blocks) or set(blocks)!=set(range(1,34)):
        raise ValueError('All 33 source entries required')
    if any(type(block) is not AlignedBlock for block in blocks.values()):
        raise ValueError('Aligned source blocks required')
    reference=blocks[1]
    seen=set()
    for ids in (reference.training_ids,reference.query_ids):
        if type(ids) is not tuple or not ids or any(type(i) is not str or not i for i in ids):
            raise ValueError('Nonempty opaque source identities required')
        if len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Duplicate source identities')
        seen.update(ids)
    folds=np.asarray(reference.folds)
    if folds.shape!=(len(reference.training_ids),) or folds.dtype.kind not in 'iu' or set(folds.tolist())!=set(range(5)):
        raise ValueError('Five aligned source folds required')
    for entry,block in blocks.items():
        if block.training_ids!=reference.training_ids or block.query_ids!=reference.query_ids:
            raise ValueError('Source block identity order differs')
        if entry==10:
            if block.folds is not None:raise ValueError('Embedding entry must remain unsupervised')
        elif not np.array_equal(block.folds,reference.folds):
            raise ValueError('Source block fold assignment differs')
        width=5 if entry==10 else 9
        for array,size in ((block.training,len(reference.training_ids)),(block.query,len(reference.query_ids))):
            if np.shape(array)!=(size,width) or not np.isfinite(array).all():
                raise ValueError('Source block shape or values differ')
    return dict(blocks)
