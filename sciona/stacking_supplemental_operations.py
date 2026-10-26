"""Draft separate supplemental producers with explicit source fold policy.

Raw class-distance calculations are shared across nearest-one/two/four outputs.
TF-IDF is learned inside each excluding fold; the declared embedding is fixed.
Clusters fit training features without labels, and both populations use that fit.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from sciona.otto_preprocessing import representation
from sciona.otto_neighbor_crossfit import crossfit_distances
from sciona.otto_tfidf_crossfit import crossfit_tfidf_distances
from sciona.otto_clustering import fit_clusters
from sciona.otto_meta_layout import AlignedBlock
from sciona.otto_tsne import PopulationEmbedding


@dataclass(frozen=True, repr=False)
class SupplementalInputs:
    args: tuple
    embedded_training: np.ndarray
    embedded_query: np.ndarray
    seed: int
    raw_metrics: tuple
    tfidf_metrics: tuple
    embedding_metrics: tuple
    tfidf_controls: dict
    cluster_controls: dict


def prepare_supplemental(training, labels, folds, training_ids, query, query_ids,
                         embedding, seed, raw_metrics, tfidf_metrics, embedding_metrics,
                         tfidf_controls, cluster_controls):
    x=representation(training,kind='raw');q=representation(query,kind='raw')
    y=np.array(labels,copy=True);f=np.array(folds,copy=True)
    if q.shape[1]!=x.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):
        raise ValueError('Five aligned folds required')
    seen=set()
    for ids,size in ((training_ids,len(x)),(query_ids,len(q))):
        if type(ids) is not list or len(ids)!=size or any(type(i) is not str or not i for i in ids):
            raise ValueError('Aligned opaque row identities required')
        if len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Overlapping source identities')
        seen.update(ids)
    if any(np.sum((f!=fold)&(y==label))<4 for fold in range(5) for label in range(9)):
        raise ValueError('Each fitting fold requires four references per class')
    if not isinstance(embedding,PopulationEmbedding):raise ValueError('Fixed population embedding required')
    for metrics,allowed in ((raw_metrics,('euclidean','cityblock','braycurtis')),
                            (tfidf_metrics,('euclidean','cityblock','braycurtis')),
                            (embedding_metrics,('euclidean','cityblock'))):
        if type(metrics) is not list or not metrics or any(type(m) is not str or m not in allowed for m in metrics) or len(set(metrics))!=len(metrics):
            raise ValueError('Distinct explicit supported metrics required')
    if type(tfidf_controls) is not dict or set(tfidf_controls)!={'smooth_idf','sublinear_tf','norm'}:
        raise ValueError('Explicit TF-IDF controls required')
    if type(cluster_controls) is not dict or set(cluster_controls)!={'kind','standardize','ddof','cluster_counts','n_init','max_iter'}:
        raise ValueError('Explicit clustering controls required')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Signed integer seed required')
    z=embedding.lookup(training_ids);v=embedding.lookup(query_ids)
    if z.shape!=(len(x),3) or v.shape!=(len(q),3) or not np.isfinite(z).all() or not np.isfinite(v).all():
        raise ValueError('Aligned finite three-dimensional coordinates required')
    for array in (x,q,y,f,z,v):array.setflags(write=False)
    return SupplementalInputs((x,y,f,list(training_ids),q,list(query_ids)),z,v,seed,
        tuple(raw_metrics),tuple(tfidf_metrics),tuple(embedding_metrics),deepcopy(tfidf_controls),deepcopy(cluster_controls))


def _args(inputs):
    if type(inputs) is not SupplementalInputs:raise ValueError('Prepared supplemental inputs required')
    return inputs.args


def _block(inputs,a,b,supervised):
    x,y,f,ids,q,qids=_args(inputs)
    a=np.asarray(a);b=np.asarray(b)
    if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=len(x) or b.shape[0]!=len(q) or a.shape[1]<1 or a.shape[1]!=b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite supplemental matrices required')
    return AlignedBlock(a,b,tuple(ids),tuple(qids),f.copy() if supervised else None)


def raw_class_distances(inputs):
    args=_args(inputs)
    raw=[crossfit_distances(*args,metric=metric) for metric in inputs.raw_metrics]
    return tuple(_block(inputs,np.column_stack([r['oof'][:,entry,:] for r in raw]),
                        np.column_stack([r['query'][:,entry,:] for r in raw]),True) for entry in range(3))


def weighted_class_distances(inputs):
    args=_args(inputs)
    values=[crossfit_tfidf_distances(*args,metric=metric,**inputs.tfidf_controls) for metric in inputs.tfidf_metrics]
    return _block(inputs,np.column_stack([r['oof'] for r in values]),np.column_stack([r['query'] for r in values]),True)


def embedded_class_distances(inputs):
    x,y,f,ids,q,qids=_args(inputs)
    values=[crossfit_distances(inputs.embedded_training,y,f,ids,inputs.embedded_query,qids,metric=metric)
            for metric in inputs.embedding_metrics]
    return _block(inputs,np.column_stack([r['oof'][:,0,:] for r in values]),
                  np.column_stack([r['query'][:,0,:] for r in values]),True)


def fit_reference_clusters(inputs):
    return fit_clusters(_args(inputs)[0],seed=inputs.seed,**inputs.cluster_controls)


def cluster_assignments(inputs,model):
    x,y,f,ids,q,qids=_args(inputs)
    return _block(inputs,model.transform(x)['assignments'],model.transform(q)['assignments'],False)


def nonzero_counts(inputs):
    x,y,f,ids,q,qids=_args(inputs)
    return _block(inputs,np.count_nonzero(x,axis=1)[:,None],np.count_nonzero(q,axis=1)[:,None],False)


def raw_neural_features(inputs):
    x,y,f,ids,q,qids=_args(inputs)
    return _block(inputs,x,q,False)


def collect_supplemental(raw_one,raw_two,raw_four,weighted,embedded,clusters,counts,raw_neural):
    values=(raw_one,raw_two,raw_four,weighted,embedded,clusters,counts,raw_neural)
    if any(type(value) is not AlignedBlock for value in values):
        raise ValueError('All aligned supplemental blocks required')
    ids,qids=raw_one.training_ids,raw_one.query_ids
    seen=set()
    for identities in (ids,qids):
        if type(identities) is not tuple or not identities or any(type(i) is not str or not i for i in identities):
            raise ValueError('Nonempty opaque identities required')
        if len(set(identities))!=len(identities) or seen.intersection(identities):
            raise ValueError('Duplicate supplemental identities')
        seen.update(identities)
    f=np.asarray(raw_one.folds)
    if f.shape!=(len(ids),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):
        raise ValueError('Five aligned supplemental folds required')
    for index,value in enumerate(values):
        if value.training_ids!=ids or value.query_ids!=qids:raise ValueError('Supplemental identity order differs')
        if index<5:
            if not np.array_equal(value.folds,f):raise ValueError('Supplemental fold assignment differs')
        elif value.folds is not None:raise ValueError('Unsupervised feature carries fold labels')
        a,b=np.asarray(value.training),np.asarray(value.query)
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=len(ids) or b.shape[0]!=len(qids) or a.shape[1]<1 or a.shape[1]!=b.shape[1]:
            raise ValueError('Supplemental block shapes differ')
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
            raise ValueError('Finite nonnegative supplemental values required')
        if index<5 and a.shape[1]%9:raise ValueError('Nine-class distance groups required')
        if index==6 and a.shape[1]!=1:raise ValueError('One nonzero-count column required')
    return dict(zip(range(1,8),values[:7])),raw_neural


def build_decomposed(training,labels,folds,training_ids,query,query_ids,*,embedding,seed,
                     raw_metrics,tfidf_metrics,embedding_metrics,tfidf_controls,cluster_controls):
    inputs=prepare_supplemental(training,labels,folds,training_ids,query,query_ids,embedding,seed,
                               raw_metrics,tfidf_metrics,embedding_metrics,tfidf_controls,cluster_controls)
    blocks=dict(zip((1,2,3),raw_class_distances(inputs)))
    blocks[4]=weighted_class_distances(inputs)
    blocks[5]=embedded_class_distances(inputs)
    blocks[6]=cluster_assignments(inputs,fit_reference_clusters(inputs))
    blocks[7]=nonzero_counts(inputs)
    return dict(supplemental=blocks,raw_neural=raw_neural_features(inputs))
