"""Independent inference using pinned preparation, tiling, crops and networks."""
import ast
import textwrap
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from scipy.ndimage import zoom,binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from scripts.audit_competition_dsb_semantics import checked_source
from scripts.dsb_source_preparation import SourcePreparation,ArrayPreparation
from scripts.validate_dsb_network import reference


class SourceInference:
    def __init__(self,source_root,pins):
        self.preparation=SourcePreparation(source_root,pins)
        tree=ast.parse(checked_source(source_root,pins,'preprocessing/full_prep.py'))
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['process_mask','lumTrans','resample','savenpy']]
        self.prepcode=compile(ast.fix_missing_locations(ArrayPreparation().visit(tree)),'<source-inference-preparation>','exec')
        tree=ast.parse(checked_source(source_root,pins,'split_combine.py'))
        tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        for node in ast.walk(tree):
            if isinstance(node,ast.AugAssign) and isinstance(node.op,ast.Div):node.op=ast.FloorDiv()
        scope={'np':np};exec(compile(tree,'<source-split-combine>','exec'),scope)
        self.split=scope['SplitComb']
        source=checked_source(source_root,pins,'data_detector.py')
        start=source.index('            nz, nh, nw = imgs.shape[1:]')
        stop=source.index('            return torch.from_numpy(imgs.astype',start)
        tree=ast.parse(textwrap.dedent(source[start:stop]))
        for node in ast.walk(tree):
            if (isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.right,ast.Attribute)
                    and node.right.attr=='stride' and not isinstance(node.left,ast.Call)):node.op=ast.FloorDiv()
        self.tilecode=compile(tree,'<source-inference-tiles>','exec')
        tree=ast.parse(checked_source(source_root,pins,'layers.py'))
        tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['GetPBB','nms','iou']]
        scope={'np':np};exec(compile(tree,'<source-proposals>','exec'),scope)
        self.decode=scope['GetPBB']({'stride':4,'anchors':[10.,30.,60.]});self.nms=scope['nms']
        source=checked_source(source_root,pins,'data_classifier.py')
        tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='simpleCrop']
        for node in ast.walk(tree):
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and (
                isinstance(node.left,ast.Name) and node.left.id=='crop_size' or
                isinstance(node.right,ast.Attribute) and node.right.attr=='stride'):node.op=ast.FloorDiv()
        scope=dict(np=np,zoom=zoom,warnings=warnings);exec(compile(tree,'<source-classifier-crop>','exec'),scope)
        self.crop=scope['simpleCrop']
        start=source.index('        croplist = np.zeros');stop=source.index("        if self.phase!='test':",start)
        tree=ast.parse(textwrap.dedent(source[start:stop]))
        for node in ast.walk(tree):
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.right,ast.Attribute) and node.right.attr=='stride':node.op=ast.FloorDiv()
        self.batchcode=compile(tree,'<source-classifier-batch>','exec')
        self.detector=reference(source_root,pins,'net_detector.py',{'Net'})['Net']
        self.classifier=reference(source_root,pins,'net_classifier.py',{'Net','CaseNet'})['CaseNet']

    def preprocess(self,volume,spacing,requested_spacing):
        saved=[]
        proxy=SimpleNamespace(**{key:getattr(np,key) for key in dir(np) if not key.startswith('_')})
        proxy.save=lambda _,value:saved.append(value.copy())
        scope=dict(np=proxy,zoom=zoom,warnings=warnings,binary_dilation=binary_dilation,
            generate_binary_structure=generate_binary_structure,convex_hull_image=convex_hull_image,
            os=SimpleNamespace(path=SimpleNamespace(join=lambda *args:'synthetic')),
            requested_resolution=np.asarray(requested_spacing),
            step1_python=lambda _:self.preparation.segment((volume.copy(),np.asarray(spacing))))
        exec(self.prepcode,scope)
        scope['savenpy'](0,['synthetic'],'synthetic','synthetic',use_existing=False)
        assert len(saved)==2
        return saved[0]

    def predict(self,volume,detector_state,classifier_state,*,topk=5,crop_size=96,tile_side=144,margin=32,tile_batch_size=1):
        detector=self.detector().eval();detector.load_state_dict(detector_state,strict=True)
        classifier=self.classifier(topk).eval();classifier.load_state_dict(classifier_state,strict=True)
        split=self.split(tile_side,16,4,margin,170)
        scope=dict(np=np,imgs=volume.copy(),self=SimpleNamespace(stride=4,pad_value=170,split_comber=split))
        exec(self.tilecode,scope)
        outputs=[]
        with torch.inference_mode():
            for start in range(0,len(scope['imgs']),tile_batch_size):
                x=torch.from_numpy(scope['imgs'][start:start+tile_batch_size].astype(np.float32))
                coords=torch.from_numpy(scope['coord2'][start:start+tile_batch_size].astype(np.float32))
                outputs.append(detector(x,coords).numpy())
            combined=split.combine(np.concatenate(outputs),nzhw=scope['nzhw'])
            proposals=self.decode(combined,thresh=-3.)
            proposals=self.nms(proposals[proposals[:,0]>-1.],.05)
            chosen=proposals[:,0].argsort()[::-1][:topk]
            config=dict(crop_size=[crop_size]*3,scaleLim=[.85,1.15],radiusLim=[6,100],jitter_range=.15,
                        augtype={'scale':False},stride=4,filling_value=160)
            batch=dict(np=np,img=volume,pbb=proposals,pbb_label=np.zeros(len(proposals)),chosenid=chosen,topk=topk,
                self=SimpleNamespace(topk=topk,crop_size=[crop_size]*3,stride=4,phase='test',crop=self.crop(config,'test')))
            exec(self.batchcode,batch)
            _,case,each=classifier(torch.from_numpy(batch['croplist'][None]),torch.from_numpy(batch['coordlist'][None]))
        return dict(case_probability=case.numpy(),proposal_probabilities=each.numpy(),proposals=proposals)
