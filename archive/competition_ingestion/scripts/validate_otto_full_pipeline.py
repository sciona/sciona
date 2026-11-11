"""Run the complete Otto synthetic pipeline; no catalog writes."""
import argparse
import json
from pathlib import Path
import numpy as np
from sciona.otto_first_level import build
from sciona.otto_tsne import fit_population
from sciona.otto_supplemental import build as supplemental
from sciona.otto_meta_layout import assemble
from sciona.otto_meta_stage import fit as fit_meta_stage


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--r-library',required=True);parser.add_argument('--libfm',required=True);parser.add_argument('--output',required=True)
    cli=parser.parse_args()
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),150);f=np.arange(len(y))%5
    x=np.column_stack((np.eye(9)[y]*20,np.ones((len(y),4))))+rng.poisson(.3,size=(len(y),13))
    q=np.array([x[y==i].mean(axis=0) for i in range(9)])
    ids=[f'r{i}' for i in range(len(x))];qids=[f'q{i}' for i in range(len(q))]
    e=fit_population(np.vstack((x,q)),ids+qids,seed=12,perplexity=20,learning_rate=50,max_iter=350)
    forest=dict(ntree=32,mtry=4,nodesize=1)
    boost=dict(rounds=8,max_depth=3,eta=.4,subsample=.8,colsample_bytree=.8)
    cluster=dict(cluster_counts=[2,3],ddof=1,n_init=2,max_iter=100)
    neural=dict(hidden=[16],epochs=[20]*2,batch_size=64,learning_rate=.1,momentum=.9,ddof=1)
    c={1:dict(forest,r_library=cli.r_library),2:dict(C=1.,max_iter=2000),3:dict(n_estimators=32,max_features=4,min_samples_leaf=1),4:dict(neighbors=8,metric='euclidean',ddof=1),5:dict(executable=cli.libfm,factors=4,iterations=40,learn_rate=.05,regularization=[0.,0.,.01]),6:dict(hidden=[16],epochs=30,activation='Rectifier',standardize=True),7:dict(alpha=1.),8:neural,9:dict(neural,epochs=[20]*6),10:dict(cluster_counts=[2,3],n_init=2,max_iter=100),11:dict(regularization=.01,iterations=10000,ddof=1,r_library=cli.r_library),12:dict(forest=forest,triples=[(0,1,2),(3,4,5)],ddof=1,r_library=cli.r_library,sofia=dict(regularization=.01,iterations=10000)),13:dict(forest=forest,triples=[(0,1,2),(3,4,5)],log_scope='nonnegative_blocks',r_library=cli.r_library,sofia=dict(regularization=.01,iterations=10000,rank_probability=.5)),14:boost,15:dict(augmentation=dict(cluster,cluster_counts=list(range(2,9))),boosting=boost),21:boost,22:dict(neighbors=8,metric='euclidean',ddof=None),23:dict(neighbors=8,metric='euclidean',ddof=None)}
    for i in (16,17,18):c[i]=dict(clustering=cluster,boosting=boost)
    for i,depth in ((19,2),(20,3)):c[i]=dict(neural,hidden=[16]*depth,epochs=[20,21,22]*40,representation='log1p')
    for i in range(24,34):c[i]=dict(metric='euclidean')
    first=build(x,y,f,ids,q,qids,embedding=e,seed=12,controls=c,progress=lambda entry:print('Validated source entry',entry,flush=True))
    extra=supplemental(x,y,f,ids,q,qids,embedding=e,seed=12,raw_metrics=['euclidean'],tfidf_metrics=['euclidean'],embedding_metrics=['euclidean'],tfidf_controls=dict(smooth_idf=True,sublinear_tf=False,norm='l2'),cluster_controls=dict(cluster,kind='raw',standardize=False))
    layout=assemble(first,extra['supplemental'],extra['raw_neural'],ids,qids,f,tsne_interpretation='five_features')
    assert len(first)==33 and len(extra['supplemental'])==7
    assert layout['tree_training'].shape[0]==len(x)
    assert layout['neural_training'].shape[1]==layout['tree_training'].shape[1]+x.shape[1]
    print('Complete first-level layout assembled; starting meta selection',flush=True)
    candidates={
        'xgboost':[dict(boost,rounds=rounds) for rounds in (4,8)],
        'adaboost_extratrees':[dict(algorithm='SAMME',boost_rounds=4,learning_rate=.7,trees=3,max_depth=depth,min_samples_leaf=2,max_features=9) for depth in (1,3)],
        'lasagne_neural':[dict(hidden=[16],epochs=epochs,batch_size=64,learning_rate=.1,momentum=.9,ddof=1) for epochs in (1,20)]}
    complete=fit_meta_stage(layout,y,np.arange(len(y))%4,ids,seed=12,candidates=candidates,progress=lambda family:print('Validated meta selection and refit',family,flush=True))
    probabilities=np.asarray(complete['blend']['probabilities'])
    assert probabilities.shape==(9,9) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1),1.)
    np.testing.assert_array_equal(complete['blend']['classes'],np.arange(9))
    result=dict(status='passed',approved=False,source_entries_executed=33,supplemental_blocks_executed=7,synthetic_training_rows=len(x),synthetic_query_rows=len(q),first_level_columns=293,tree_columns=layout['tree_training'].shape[1],neural_columns=layout['neural_training'].shape[1],gpu_validated=False,meta_selection=complete['selection'],synthetic_query_correct=9,catalog_mutations=0)
    Path(cli.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: complete first-level bank, meta selection/refits and final blend',flush=True)


if __name__=='__main__':main()
