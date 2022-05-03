import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time
from torchvision.ops import nms
from objectDet3D.networks.utils import DETECTOR_DICT
from objectDet3D.networks.detectors.KM3D_core import KM3DCore
from objectDet3D.networks.heads.km3d_head import KM3DHead
from objectDet3D.networks.heads.monoflex_head import MonoFlexHead
from objectDet3D.networks.lib.blocks import AnchorFlatten
from objectDet3D.networks.lib.look_ground import LookGround
from objectDet3D.networks.lib.ops.dcn.deform_conv import DeformConv

@DETECTOR_DICT.register_module
class KM3D(nn.Module):
    
    def __init__(self, network_jfc):
        super(KM3D, self).__init__()

        self.obj_types = network_jfc.obj_types

        self.build_head(network_jfc)

        self.build_core(network_jfc)

        self.network_jfc = network_jfc


    def build_core(self, network_jfc):
        self.core = KM3DCore(network_jfc.backbone)

    def build_head(self, network_jfc):
        self.bbox_head = KM3DHead(
            **(network_jfc.head)
        )

    def training_forward(self, img_batch, annotations, meta):
       

        features  = self.core(dict(image=img_batch, P2=meta['P2']))
        output_dict = self.bbox_head(features)

        loss, loss_dict = self.bbox_head.loss(output_dict, annotations, meta)

        return loss, loss_dict
    
    def test_forward(self, img_batch, P2):
       
        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing

        features  = self.core(dict(image=img_batch, P2=P2))
        output_dict = self.bbox_head(features)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(output_dict, P2, img_batch)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, meta = inputs
            return self.training_forward(img_batch, annotations, meta)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch, calib)

@DETECTOR_DICT.register_module
class MonoFlex(KM3D):
    
    def build_head(self, network_jfc):
        self.bbox_head = MonoFlexHead(**(network_jfc.head))

