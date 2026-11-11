"""Source box retention for image crops and mosaic quadrants."""
import numpy as np


def crop_image_and_boxes(image, boxes, xmin, ymin, xmax, ymax):
    image=np.asarray(image)
    boxes=np.asarray(boxes)
    if image.ndim!=3 or image.shape[2]!=3 or boxes.ndim!=2 or boxes.shape[1]!=4:
        raise ValueError('RGB image and xyxy box matrix required')
    if not all(type(v) is int for v in (xmin,ymin,xmax,ymax)) or not (0<=xmin<xmax<=image.shape[1] and 0<=ymin<ymax<=image.shape[0]):
        raise ValueError('nonempty in-image integer crop required')
    retained=[]
    for box in boxes:
        width=max(0,min(box[2],xmax)-max(box[0],xmin)+1)
        height=max(0,min(box[3],ymax)-max(box[1],ymin)+1)
        area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
        if not np.isfinite(box).all() or area<=0:
            raise ValueError('finite positive-area source boxes required')
        if width*height/float(area)>.25:
            retained.append(box)
    result=np.asarray(retained,dtype=float).reshape(-1,4)
    result[:,[0,2]]-=xmin
    result[:,[1,3]]-=ymin
    result[:,[0,2]]=result[:,[0,2]].clip(0,xmax-xmin)
    result[:,[1,3]]=result[:,[1,3]].clip(0,ymax-ymin)
    return image[ymin:ymax,xmin:xmax,:],result
