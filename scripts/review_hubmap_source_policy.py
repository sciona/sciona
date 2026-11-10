"""Extract only model policy from hash-pinned public HuBMAP software."""
import argparse
import ast
import json
from pathlib import Path
from scripts.audit_hubmap_original_readiness import ROOT,sha
from scripts.plasticc_publication_evidence import strict_equal

TRAINING=['src/02_train/get_config.py','src/05_train_with_pseudo_labels/get_config.py']
PSEUDO_CONFIG='src/03_generate_pseudo_labels/03_01_pseudo_label_kaggle_data/get_config.py'
PSEUDO_SCRIPT='src/03_generate_pseudo_labels/03_01_pseudo_label_kaggle_data/pseudo_label_kaggle_data_03_01.py'
TRAIN_KEYS=['split_seed_list','FOLD_LIST','input_resolution','resolution','clf_threshold','tta',
            'trn_batch_size','test_batch_size','num_epochs','early_stopping','patience']
PSEUDO_KEYS=['split_seed_list','FOLD_LIST','input_resolution','resolution','clf_threshold','tta','pad_size','test_batch_size']


def extract_config(source,keys):
    matches=[]
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node,ast.Dict):continue
        entries={k.value:v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and isinstance(k.value,str)}
        if 'FOLD_LIST' in entries:
            matches.append({key:ast.literal_eval(entries[key]) for key in keys})
    if len(matches)!=1:raise ValueError('Unique top-level model policy required')
    return json.loads(json.dumps(matches[0]))


def extract_checkpoint_role(source):
    tree=ast.parse(source)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='seed'
           and ast.unparse(n.iter)=="config['split_seed_list']"]
    selections=[]
    for loop in loops:
        for node in loop.body:
            if isinstance(node,ast.For) and ast.unparse(node.target)=='fold' and ast.unparse(node.iter)=="config['FOLD_LIST']":
                selections.extend(ast.unparse(x) for x in ast.walk(node) if isinstance(x,ast.JoinedStr))
    if selections!=["f'model_seed{seed}_fold{fold}_bestscore.pth'"]:
        raise ValueError('Source checkpoint selection differs')
    return 'best_score'


def review(source_root):
    pins={}
    pin_hashes={}
    for suffix in ['source','pseudo']:
        path=ROOT/('docs/reviews/competition_hubmap_'+suffix+'_pins.json')
        record=json.loads(path.read_text())
        if record['commit']!='615444e86f4fb5ab916c0b66d311a307f4e4dd27':raise ValueError('Pinned source commit differs')
        pins.update(record['files']);pin_hashes[str(path.relative_to(ROOT))]=sha(path)
    configs=sorted(n for n in pins if '/03_generate_pseudo_labels/' in n and n.endswith('/get_config.py'))
    scripts=sorted(n for n in pins if '/03_generate_pseudo_labels/' in n and '/pseudo_label_' in n)
    if len(configs)!=3 or len(scripts)!=3:raise ValueError('Three source pseudo branches required')
    paths=TRAINING+configs+scripts
    for name in paths:
        if sha(source_root/name)!=pins[name]:raise ValueError('Pinned public software changed')
    training=[extract_config((source_root/name).read_text(),TRAIN_KEYS) for name in TRAINING]
    expected=dict(split_seed_list=[0],FOLD_LIST=[0,1,2,3],input_resolution=[320,320],resolution=[1024,1024],
                  clf_threshold=None,tta=1,trn_batch_size=16,test_batch_size=16,num_epochs=20,early_stopping=True,patience=50)
    if any(not strict_equal(r,expected) for r in training):raise ValueError('Source training policy differs')
    policies=[extract_config((source_root/name).read_text(),PSEUDO_KEYS) for name in configs]
    wanted=dict(split_seed_list=[0],FOLD_LIST=[0,1,2,3],input_resolution=320,resolution=1024,
                clf_threshold=.5,tta=4,pad_size=256,test_batch_size=12)
    if any(not strict_equal(p,wanted) for p in policies):raise ValueError('Source pseudo-label policy differs')
    roles=[extract_checkpoint_role((source_root/name).read_text()) for name in scripts]
    return dict(format='hubmap-source-policy.v2',passed=True,approved=False,catalog_mutations=0,
        source_commit='615444e86f4fb5ab916c0b66d311a307f4e4dd27',
        initial_training=training[0],retraining=training[1],pseudo_inference=policies[0],pseudo_checkpoint_role=roles[0],
        pseudo_branches_verified=3,all_pseudo_branch_policies_equal=True,
        source_software_sha256={name:pins[name] for name in paths},pin_manifest_sha256=pin_hashes,
        reviewer_sha256=sha(Path(__file__)),
        limits=['Static extraction of model policy only; no runtime execution, historical weight authentication or predictive-quality claim.',
                'Final-notebook checkpoint selection is separate and is not inferred from pseudo-label selection.',
                'Only model policy is exported. Source data paths, identifiers and runtime records are excluded.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',required=True,type=Path)
    report=review(parser.parse_args().source_root)
    (ROOT/'docs/reviews/hubmap_source_policy.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,folds=4,epochs=20,patience=50,pseudo_checkpoint_role=report['pseudo_checkpoint_role'],approved=False)))
