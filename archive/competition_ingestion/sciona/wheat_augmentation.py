"""Full source photometric/flip policy and box-aware detector resizing."""
import numpy as np


class WheatAugmentation:
    """Construct and execute within historical_augmentation_api in an isolated worker."""
    def __init__(self, image_size):
        import albumentations as A
        import imgaug
        if A.__version__!='0.4.5' or imgaug.__version__!='0.2.6':
            raise ValueError('qualified historical augmentation packages required')
        if image_size not in (512,640,768,1024):
            raise ValueError('source detector image size required')
        def compose(transforms):
            return A.Compose(transforms,bbox_params=A.BboxParams(format='pascal_voc',min_area=0,
                min_visibility=0,label_fields=['category_id']))
        self.train_transforms=compose([
            A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5),A.ToGray(p=.01),
            A.OneOf([A.IAAAdditiveGaussianNoise(),A.GaussNoise()],p=.2),
            A.OneOf([A.MotionBlur(p=.2),A.MedianBlur(blur_limit=3,p=.1),A.Blur(blur_limit=3,p=.1)],p=.2),
            A.OneOf([A.CLAHE(),A.IAASharpen(),A.IAAEmboss(),A.RandomBrightnessContrast()],p=.25),
            A.HueSaturationValue(p=.25)])
        self.resize_transforms=compose([A.Resize(height=image_size,width=image_size,interpolation=1,p=1)])

    @staticmethod
    def _apply(transform,image,boxes):
        result=transform(image=image,bboxes=boxes,category_id=np.ones(len(boxes),dtype=int))
        return result['image'],np.array(result['bboxes'])

    def resize(self,image,boxes):
        return self._apply(self.resize_transforms,image,boxes)

    def augment(self,image,boxes):
        return self._apply(self.train_transforms,image,boxes)
