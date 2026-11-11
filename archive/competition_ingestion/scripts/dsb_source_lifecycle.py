"""Reusable pinned source training/validation loop over independent source loaders."""
import ast
import time
from types import SimpleNamespace
import numpy as np
import torch
from scripts.audit_competition_dsb_semantics import checked_source
from scripts.validate_dsb_epoch import SourceCompatibility,SourceSGD,training_function
from scripts.validate_dsb_validation import source_function
from scripts.validate_dsb_network import reference


class ReopenedLoader:
    def __init__(self,open_batches):self.open_batches=open_batches
    def __iter__(self):return iter(self.open_batches())


class SourceLifecycle:
    def __init__(self,source_root,pins,variant):
        read=lambda name:checked_source(source_root,pins,'training/classifier/'+name)
        self.feature=reference(source_root,pins,'net_classifier.py',{'Net'})['Net']
        self.case=reference(source_root,pins,f'training/classifier/net_classifier_{variant}.py',{'CaseNet'})['CaseNet']
        self.train_detector=training_function(read('trainval_detector.py'),'train_nodulenet')
        self.train_classifier=training_function(read('trainval_classifier.py'),'train_casenet')
        scope=dict(np=np,torch=torch,nn=torch.nn,time=time,Variable=lambda value,**kwargs:value,
                   binary_cross_entropy=torch.nn.functional.binary_cross_entropy)
        tree=ast.parse(read('layers.py'))
        tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Loss','hard_mining']]
        exec(compile(ast.fix_missing_locations(SourceCompatibility().visit(tree)),'<source-loss>','exec'),scope)
        self.loss=scope['Loss'](2)
        self.val_detector=source_function(read('trainval_detector.py'),'validate_nodulenet',scope)
        self.val_classifier=source_function(read('trainval_classifier.py'),'val_casenet',scope)
        main=ast.parse(read('main.py'))
        loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
        self.control=compile(ast.Module(body=loop.body[:3],type_ignores=[]),'<source-epoch-control>','exec')
        self.configs={}
        for role,name in [('detector','net_detector_3.py'),('classifier',f'net_classifier_{variant}.py')]:
            tree=ast.parse(read(name));nodes=[]
            for node in tree.body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript):
                    target=node.targets[0]
                    if isinstance(target.value,ast.Name) and target.value.id=='config' and isinstance(target.slice,ast.Constant) and target.slice.value in ['augtype','lr_stage','lr','startepoch']:
                        nodes.append(node)
            scope={'np':np,'config':{}}
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-safe-config>','exec'),scope)
            self.configs[role]=scope['config']

    def adapt(self,detector_state,topk):
        feature=self.feature();feature.load_state_dict(detector_state,strict=True)
        return self.case(topk,feature)

    def restore(self,classifier_state,topk):
        feature=self.feature();model=self.case(topk,feature)
        model.load_state_dict(classifier_state,strict=True)
        optimizers=[SourceSGD(m.parameters(),lr=0.,momentum=.9,weight_decay=.0001) for m in [feature,model]]
        return model,optimizers

    def run(self,model,optimizers,training,validation,*,start,end,freeze=False):
        detector_config=self.configs['detector'];classifier_config=self.configs['classifier']
        args=SimpleNamespace(lr=None,debug=False,freeze_batchnorm=freeze,
            lr_stage=detector_config['lr_stage'],lr_preset=detector_config['lr'],
            lr_stage2=classifier_config['lr_stage'],lr_preset2=classifier_config['lr'],miss_thresh=.03,miss_ratio=1.)
        train_det=ReopenedLoader(lambda:training('detector',detector_config['augtype']))
        train_case=ReopenedLoader(lambda:training('classifier',classifier_config['augtype']))
        val_det=ReopenedLoader(lambda:validation('detector'))
        val_case=ReopenedLoader(lambda:validation('classifier_validation'))
        val_all=ReopenedLoader(lambda:validation('classifier_training_validation'))
        observed=[]
        def validate_detector(batches,net,loss):
            with torch.inference_mode():value=self.val_detector(batches,net,loss)
            observed.append(('detector',value['metrics']))
        def validate_classifier(epoch,net,batches,args):
            with torch.inference_mode():value=self.val_classifier(epoch,net,batches,args)
            observed.append(('classifier_validation' if batches is val_case else 'classifier_training_validation',value))
        controls=dict(start_epoch=start,args=args,config2=classifier_config,case_net=model,nod_net=model.NoduleNet,
            loss=self.loss,optimizer=optimizers[0],optimizer2=optimizers[1],train_loader_case=train_case,
            train_loader_nod=train_det,val_loader_nod=val_det,val_loader_case=val_case,all_loader_case=val_all,
            train_casenet=self.train_classifier,train_nodulenet=self.train_detector,
            validate_nodulenet=validate_detector,val_casenet=validate_classifier)
        for epoch in range(start,end+1):controls['epoch']=epoch;exec(self.control,controls)
        return observed
