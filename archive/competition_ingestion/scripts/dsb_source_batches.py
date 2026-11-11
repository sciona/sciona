"""Source-only batch construction for the independent DSB execution reference.

Uses pinned loader methods/helpers, explicit in-memory collections and orders.
No candidate sample, crop, augmentation, label or sampling callable is executed.
"""
import ast
from types import SimpleNamespace
import numpy as np
import torch
from scripts.audit_competition_dsb_semantics import checked_source
from scripts.validate_dsb_validation_samples import reference


class SourceTrainingBatches:
    def __init__(self,source_root,pins,arrays,options):
        self.arrays=arrays;self.options=options
        self.scopes={role:reference(checked_source(source_root,pins,'training/classifier/data_'+role+'.py'),
            role,options['numpy_rng'],options['label_rng'],None) for role in ['detector','classifier']}
        # Keep the literal source filtered-position bug out of reference cases.
        # The separately tested candidate correction is not source equivalence.
        if list(arrays['eligible_image_indices'])!=list(range(len(arrays['detector_volumes']))):
            raise ValueError('source reference requires identity eligible-image mapping')
        detector_tree=ast.parse(checked_source(source_root,pins,'training/classifier/data_detector.py').expandtabs(8))
        cls=next(n for n in detector_tree.body if isinstance(n,ast.ClassDef) and n.name=='DataBowl3Detector')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        start=next(i for i,n in enumerate(init.body) if ast.unparse(n)=='self.bboxes = []')
        subject=SimpleNamespace();resolution=options.get('resolution',1.)
        exec(compile(ast.Module(body=init.body[start:start+3],type_ignores=[]),'<source-table>','exec'),
            dict(np=np,self=subject,labels=arrays['boxes_by_image'],sizelim=6/resolution,sizelim2=30/resolution,sizelim3=40/resolution))
        self.table=subject.bboxes

    def __call__(self,task,profile,*,phase='train'):
        scope=self.scopes[task];arrays=self.arrays;options=self.options
        volumes=arrays['detector_volumes' if task=='detector' else 'classifier_volumes']
        scope['np'].load=lambda index:volumes[index].copy()
        if task=='detector':
            config=dict(crop_size=[options.get('detector_crop_size',128)]*3,bound_size=12,stride=4,
                pad_value=170,num_neg=800,th_neg=.02,anchors=[10.,30.,60.],th_pos_train=.5,th_pos_val=1.)
            subject=SimpleNamespace(phase=phase,bboxes=self.table,filenames=list(range(len(volumes))),
                kagglenames=list(arrays['eligible_image_indices']),sample_bboxes=arrays['boxes_by_image'],
                augtype=profile,crop=scope['Crop'](config),label_mapping=scope['LabelMapping'](config,phase))
        else:
            side=options.get('classifier_crop_size',96)
            config=dict(crop_size=[side]*3,scaleLim=[.85,1.15],radiusLim=[6,100],jitter_range=.15,
                augtype=profile,stride=4,filling_value=160)
            subject=SimpleNamespace(candidate_box=arrays['proposals_by_case'],pbb_label=arrays['known_by_case'],
                T=1,topk=options.get('topk',5),filenames=list(range(len(volumes))),random_sample=True,
                phase=phase,crop_size=[side]*3,stride=4,augtype=profile,
                crop=scope['simpleCrop'](config,phase),yset=arrays['case_labels'])
        pending=[];size=options[task+'_batch_size']
        for index in options[task+'_order']:
            pending.append(scope['__getitem__'](subject,index))
            if len(pending)==size:
                yield tuple(torch.stack([torch.as_tensor(value) for value in values]) for values in zip(*pending))
                pending=[]
        if pending:yield tuple(torch.stack([torch.as_tensor(value) for value in values]) for values in zip(*pending))


class SourceValidationBatches:
    def __init__(self,source_root,pins,arrays,options):
        layers=ast.parse(checked_source(source_root,pins,'training/classifier/layers.py'))
        scope={'np':np}
        exec(compile(ast.Module(body=[n for n in layers.body if isinstance(n,ast.FunctionDef)
            and n.name in ['iou','nms']],type_ignores=[]),'<source-proposal-functions>','exec'),scope)
        tree=ast.parse(checked_source(source_root,pins,'training/classifier/data_classifier.py'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataBowl3Classifier')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        loop=next(n for n in init.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='idx')
        statements=[]
        for node in loop.body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name=='pbb_label' or (name=='pbb' and not (isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='load')):statements.append(node)
            elif isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='p':statements.append(node)
        assert len(statements)==4
        annotate=compile(ast.Module(body=statements,type_ignores=[]),'<source-proposal-annotation>','exec')
        self.loaders={}
        detector=arrays['detector']
        for key in ['classifier_validation','classifier_training_validation']:
            data=arrays[key];proposals=[];known=[]
            for p,b in zip(data['proposals'],data['boxes']):
                state={**scope,'pbb':p.copy(),'lbb':b,'config':dict(conf_th=-1.,nms_th=.05,detect_th=.05)}
                exec(annotate,state);proposals.append(state['pbb']);known.append(np.array(state['pbb_label']))
            inputs=dict(detector_volumes=detector['volumes'],boxes_by_image=detector['boxes'],
                eligible_image_indices=list(range(len(detector['volumes']))),classifier_volumes=data['volumes'],
                proposals_by_case=proposals,known_by_case=known,case_labels=data['labels'])
            loader_options={**options,'classifier_order':list(range(len(data['volumes'])))}
            loader=SourceTrainingBatches(source_root,pins,inputs,loader_options)
            loader.options['detector_order']=list(range(len(loader.table)))
            self.loaders[key]=loader
    def __call__(self,task):
        profile=dict(flip=False,rotate=False,swap=False,scale=False)
        if task=='detector':return self.loaders['classifier_validation']('detector',profile,phase='val')
        return self.loaders[task]('classifier',profile,phase='val')
