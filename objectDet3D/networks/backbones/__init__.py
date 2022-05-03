from .resnet import resnet101, resnet152, resnet18, resnet34, resnet50, ResNet, resnet
from .dla import dlanet
from objectDet3D.networks.utils.registry import BACKBONE_DICT

def build_backbone(jfc):
    temp_jfc = jfc.copy()
    name = ""
    if 'name' in temp_jfc:
        name = temp_jfc.pop('name')
    else:
        name = 'resnet'

    return BACKBONE_DICT[name](**temp_jfc)
