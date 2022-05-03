import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from objectDet3D.networks.utils.registry import DETECTOR_DICT
from objectDet3D.utils.timer import profile
from objectDet3D.networks.heads import losses
from objectDet3D.networks.detectors.yolostereo3d_core import YoloStereo3DCore
from objectDet3D.networks.heads.detection_3d_head import StereoHead
from objectDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from objectDet3D.networks.backbones.resnet import BasicBlock



@DETECTOR_DICT.register_module
class Stereo3D(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_jfc):
        super(Stereo3D, self).__init__()

        self.obj_types = network_jfc.obj_types

        self.build_head(network_jfc)

        self.build_core(network_jfc)

        self.network_jfc = network_jfc

    def build_core(self, network_jfc):
        self.core = YoloStereo3DCore(network_jfc.backbone)

    def build_head(self, network_jfc):
        self.bbox_head = StereoHead(
            **(network_jfc.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, P2, P3, disparity=None):
        
        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)

        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None:
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)
        return cls_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        
        return scores, bboxes, cls_indexes


    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)
