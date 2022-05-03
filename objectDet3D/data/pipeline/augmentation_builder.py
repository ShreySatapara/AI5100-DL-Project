from typing import Callable, List, Union
import numpy as np
from easydict import EasyDict
from objectDet3D.networks.utils.registry import AUGMENTATION_DICT
from objectDet3D.data.kitti.kittidata import KittiObj

def build_single_augmentator(jfc:EasyDict):
    name:str = jfc.type_name
    keywords:dict = getattr(jfc, 'keywords', dict())
    return AUGMENTATION_DICT[name](**keywords)

@AUGMENTATION_DICT.register_module
class Compose(object):
   

    def __init__(self, aug_list:List[EasyDict], is_return_all=True):
        self.transforms:List[Callable] = []
        for item in aug_list:
            self.transforms.append(build_single_augmentator(item))
        self.is_return_all = is_return_all

    @classmethod
    def from_transforms(cls, transforms:List[Callable]): 
        instance:Compose = cls(aug_list=[])
        instance.transforms = transforms
        return instance

    def __call__(self, left_image:np.ndarray,
                       right_image:Union[None, np.ndarray]=None,
                       p2:Union[None, np.ndarray]=None,
                       p3:Union[None, np.ndarray]=None,
                       labels:Union[None, List[KittiObj]]=None,
                       image_gt:Union[None, np.ndarray]=None,
                       lidar:Union[None, np.ndarray]=None)->List[Union[None, np.ndarray, List[KittiObj]]]:
        
        for t in self.transforms:
            left_image, right_image, p2, p3, labels, image_gt, lidar = t(left_image, right_image, p2, p3, labels, image_gt, lidar)
        return_list = [left_image, right_image, p2, p3, labels, image_gt, lidar]
        if self.is_return_all:
            return return_list
        return [item for item in return_list if item is not None]


def build_augmentator(aug_jfc:List[EasyDict])->Compose:
    return Compose(aug_jfc, is_return_all=False)
