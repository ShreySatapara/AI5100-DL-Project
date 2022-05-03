import os
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time
from typing import List
from objectDet3D.networks.utils import DETECTOR_DICT
from objectDet3D.networks.backbones import resnet
from objectDet3D.networks.lib.coordconv import CoordinateConv
from objectDet3D.networks.lib.blocks import ConvBnReLU
from objectDet3D.networks.detectors.unet.u_net import UNet_Core
from objectDet3D.networks.heads.monodepth_loss import MonodepthLoss

def preprocess_sum_avg(sum_pred:np.ndarray, num_pred:np.ndarray)->np.ndarray:
    
    return np.sum(sum_pred) / np.sum(num_pred)

def reshape_depth(gt_depth, shape):
    
    mask = gt_depth < 0.1

    inverse_gt = 1.0 / (gt_depth + 1e-9)
    inverse_gt[mask] = 1e-9
    inverse_gt_reshape = F.adaptive_max_pool2d(inverse_gt, shape)
    reshaped_gt = 1.0 / (inverse_gt_reshape + 1e-9)
    reshaped_gt[inverse_gt_reshape < 1e-8] = 0
    return reshaped_gt

@DETECTOR_DICT.register_module
class MonoDepth(nn.Module):
    
    def __init__(self, network_jfc):
        super(MonoDepth, self).__init__()
    
        self.max_depth = getattr(network_jfc, 'max_depth', 50)
        self.output_channel = getattr(network_jfc, 'output_channel', 1)
        self.backbone_arguments = getattr(network_jfc, 'backbone')
        feature_size = getattr(network_jfc, 'feature_size', 256)
        self.SI_loss_lambda = getattr(network_jfc, 'SI_loss_lambda', 0.3)
        self.smooth_weight  = getattr(network_jfc, 'smooth_loss_weight', 0.003)
        self.minor_weight  = getattr(network_jfc, 'minor_weight', 0.000)

        sum_file = os.path.join(network_jfc.preprocessed_path, 'training', 'log_depth_sum.npy')
        num_file = os.path.join(network_jfc.preprocessed_path, 'training', 'log_depth_solid.npy')
        sum_precompute = np.load(sum_file) #[H]
        num_precompute = np.load(num_file) #[H]
        
        self.register_buffer("prior_mean", torch.tensor(preprocess_sum_avg(sum_precompute, num_precompute), dtype=torch.float32))


        self.core = UNet_Core(3, self.output_channel, backbone_arguments=self.backbone_arguments)
        self.semi_loss = MonodepthLoss()

    def training_forward(self, img_batch:torch.FloatTensor, K:torch.FloatTensor, gts:torch.FloatTensor):
        

        N, C, H, W = img_batch.shape
        feat = self.core(img_batch, K)
        loss = 0
        for key in feat:
            
            depth_prediction = torch.exp(self.prior_mean + feat[key]).squeeze(1)
            shape = [depth_prediction.shape[1], depth_prediction.shape[2]]
            reshaped_gt = reshape_depth(gts, shape)
            diff = torch.log(depth_prediction) - torch.log(reshaped_gt)
            num_pixels = torch.sum((reshaped_gt > 0.1) * (reshaped_gt < self.max_depth))
            diff = torch.where(
            (reshaped_gt > 0.1) * (reshaped_gt < self.max_depth) * (torch.abs(diff) > 0.001),
            diff,
            torch.zeros_like(diff)
            )
            lamda = self.SI_loss_lambda
            loss1 = torch.sum(diff ** 2) / num_pixels - lamda * ((torch.sum(diff) / num_pixels) ** 2)

            smooth_loss = self.semi_loss.smooth_loss(feat[key], F.adaptive_avg_pool2d(img_batch, shape))

            if key == 'scale_1':
                loss += (loss1 + self.smooth_weight  * smooth_loss)
            else:
                loss += self.minor_weight * (loss1 + self.smooth_weight  * smooth_loss)
        loss_dict = dict(total_loss=loss)
        return loss, loss_dict
 
    def test_forward(self, img_batch:torch.Tensor, P2:torch.Tensor):
                
        N, C, H, W = img_batch.shape
        feat = self.core(img_batch, P2)
        

        
        depth_prediction = torch.exp(self.prior_mean + feat['scale_1'])
        
        assert(torch.all(depth_prediction > 0))

        return {"target": depth_prediction} 

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, K, gts = inputs
            return self.training_forward(img_batch, img_batch.new(K), gts)
        else:
            img_batch, K = inputs
            return self.test_forward(img_batch, K)

