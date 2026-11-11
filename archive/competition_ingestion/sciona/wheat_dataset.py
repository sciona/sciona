"""Runtime-array dataset with the winner's sampling and tensor handoff."""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from sciona.wheat_augmentation import WheatAugmentation
from sciona.wheat_crop import crop_image_and_boxes
from sciona.wheat_mosaic import mosaic_image


class WheatDataset(Dataset):
    """Use in a serial historical_augmentation_api worker with explicit RNG seeding.

    Loader returns RGB uint8 pixels, raw xyxy boxes and a source role tag.
    Training retains the source retry-until-positive policy; callers must
    provide a population from which positive examples can be sampled.
    """
    def __init__(self,image_ids,loader,image_size,mode='train',network='FasterRCNN'):
        if mode not in ('train','valid') or network not in ('FasterRCNN','EffDet'):
            raise ValueError('source dataset mode and network required')
        self.image_ids=list(np.unique(image_ids))
        self.loader=loader
        self.image_size=image_size
        self.mode,self.network=mode,network
        if mode=='train':random.shuffle(self.image_ids)
        self.transforms=WheatAugmentation(image_size)

    def __len__(self):
        return len(self.image_ids)

    def load(self,key):
        image,boxes,source=self.loader(key)
        retained=[box for box in boxes if box[2]-box[0]>=10 and box[3]-box[1]>=10]
        return image,np.array(retained,dtype=float).reshape(-1,4),source

    def random_crop_resize(self,image,boxes):
        if random.random()>.5:
            size=random.randint(768,1024)
            x=random.randint(0,1024-size)
            y=random.randint(0,1024-size)
            image,boxes=crop_image_and_boxes(image,boxes,x,y,x+size,y+size)
            return self.transforms.resize(image,boxes)
        if self.image_size!=1024:
            return self.transforms.resize(image,boxes)
        return image,boxes

    def __getitem__(self,index):
        key=self.image_ids[index]
        if self.mode=='train':
            while True:
                if random.random()>.5:
                    image,boxes,source=self.load(key)
                    if source=='spike':
                        left=0 if random.random()>.5 else image.shape[1]-1024
                        image,boxes=crop_image_and_boxes(image,boxes,left,0,left+1024,1024)
                else:
                    image,boxes=mosaic_image(key,self.image_ids,self.load,python_rng=random)
                image,boxes=self.random_crop_resize(image,boxes)
                if len(boxes)>0:
                    image,boxes=self.transforms.augment(image,boxes)
                    break
        else:
            image,boxes,_=self.load(key)
            if self.image_size!=1024:
                image,boxes=self.transforms.resize(image,boxes)
        count=len(boxes)
        if self.network=='EffDet':
            coordinates=boxes[:,[1,0,3,2]] if count else np.empty((0,4))
            target=dict(boxes=torch.as_tensor(coordinates,dtype=torch.float32),labels=torch.ones(count,dtype=torch.int64))
        else:
            coordinates=boxes if count else np.empty((0,4))
            target=dict(boxes=torch.as_tensor(coordinates,dtype=torch.float32),labels=torch.ones(count,dtype=torch.int64),
                area=torch.as_tensor((coordinates[:,3]-coordinates[:,1])*(coordinates[:,2]-coordinates[:,0]),dtype=torch.float32),
                iscrowd=torch.zeros(count,dtype=torch.int64))
        image=image.astype(np.float32)
        image/=255.
        return torch.from_numpy(image).permute(2,0,1),target
