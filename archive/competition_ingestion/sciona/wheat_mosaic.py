"""Four-image source mosaic with positional auxiliary-image cropping."""
import numpy as np

from sciona.wheat_crop import crop_image_and_boxes


def mosaic_image(image_id, image_ids, load_image_and_boxes, *, python_rng):
    """Loader supplies runtime RGB pixels, xyxy boxes and the source role tag."""
    available=list(image_ids)
    if len(available)<4 or len(set(available))!=len(available) or image_id not in available:
        raise ValueError('at least four unique source image positions required')
    available.remove(image_id)
    selected=[image_id]+python_rng.sample(available,3)
    xc,yc=[int(python_rng.uniform(256,768)) for _ in range(2)]
    result=np.full((1024,1024,3),1,dtype=np.uint8)
    parts=[]
    crops=[(1024-xc,1024-yc,1024,1024),(0,1024-yc,1024-xc,1024),
           (0,0,1024-xc,1024-yc),(1024-xc,0,1024,1024-yc)]
    destinations=[(0,0,xc,yc),(xc,0,1024,yc),(xc,yc,1024,1024),(0,yc,xc,1024)]
    for position,(key,crop,destination) in enumerate(zip(selected,crops,destinations)):
        image,boxes,source=load_image_and_boxes(key)
        if source=='spike':
            left=image.shape[1]-1024 if position in (0,3) else 0
            image,boxes=crop_image_and_boxes(image,boxes,left,0,left+1024,1024)
        image,boxes=crop_image_and_boxes(image,boxes,*crop)
        left,top,right,bottom=destination
        result[top:bottom,left:right,:]=image
        if len(boxes):
            boxes[:,[0,2]]+=left
            boxes[:,[1,3]]+=top
        parts.extend(boxes)
    boxes=np.asarray(parts,dtype=float).reshape(-1,4)
    boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,1024)
    boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,1024)
    return result,boxes
