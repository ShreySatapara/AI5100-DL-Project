import numpy as np
import torch.nn as nn
import torch
import math
import time
from torchvision.ops import nms
from objectDet3D.networks.utils import DETECTOR_DICT
from objectDet3D.networks.utils.utils import BBoxTransform, ClipBoxes
from objectDet3D.networks.heads.anchors import Anchors
from objectDet3D.networks.heads.retinanet_head import RetinanetHead
from objectDet3D.networks.heads import losses
from objectDet3D.networks.lib.blocks import ConvBnReLU
from objectDet3D.networks.backbones import resnet

class FPN(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_outs):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels[i], out_channels, 1) for i in range(len(in_channels))
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, (3, 3), padding=1) for i in range(len(in_channels))
            ]
        )

        extra_levels = num_outs - len(in_channels) # add extra down-sampling, Retinanet add convs on inputs
        if extra_levels > 0:
            for i in range(extra_levels):
                if i == 0:
                    self.fpn_convs.append(
                        nn.Conv2d(in_channels[-1], out_channels, 3, padding=1, stride=2)
                    )
                else:
                    self.fpn_convs.append(
                        nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)
                    )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        # Build Laterals
        laterals = [
            self.lateral_convs[i](feats[i]) for i in range(len(self.in_channels))
        ]

        # top-down path
        for i in range(len(self.in_channels) - 1, 0, -1):
            laterals[i - 1] += torch.nn.functional.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
        # build original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(self.in_channels))
        ]
        if len(self.fpn_convs) > len(outs):
            # RetinaNet add convolutions to inputs
            outs.append(self.fpn_convs[len(outs)](feats[-1]))

            for i in range(len(outs), len(self.fpn_convs)):
                outs.append(self.fpn_convs[i](outs[-1])) # default no relu in mmdetection retinanet, with relu in pytorch/retinanet

        return tuple(outs)

class RetinaNetCore(nn.Module):
    """Some Information about RetinaNetCore"""
    def __init__(self, backbone_jfc, neck_jfc):
        super(RetinaNetCore, self).__init__()
        self.backbone = resnet(**backbone_jfc)
        self.neck     = FPN(**neck_jfc)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return feats

@DETECTOR_DICT.register_module
class RetinaNet(nn.Module):
    
    def __init__(self, network_jfc):
        super(RetinaNet, self).__init__()

        self.clipBoxes = ClipBoxes()

        self.obj_types = network_jfc.obj_types

        self.build_core(network_jfc)

        self.build_head(network_jfc)

        self.network_jfc = network_jfc

    def build_core(self, network_jfc):
        self.core = RetinaNetCore(network_jfc.backbone, network_jfc.neck)

    def build_head(self, network_jfc):
        self.bbox_head = RetinanetHead(**(network_jfc.head) )


    def training_forward(self, img_batch, annotations):
        

        feats = self.core(img_batch)

        cls_preds, reg_preds = self.bbox_head(feats)
        anchors = self.bbox_head.get_anchor(img_batch)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations)
        return (cls_loss, reg_loss, loss_dict)
    
    def test_forward(self, img_batch):
       
        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing

        feats = self.core(img_batch)
        cls_preds, reg_preds = self.bbox_head(feats)
        anchors = self.bbox_head.get_anchor(img_batch)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, calib = inputs
            return self.training_forward(img_batch, annotations)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch)