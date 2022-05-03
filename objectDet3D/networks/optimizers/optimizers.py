from easydict import EasyDict 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def build_optimizer(optim_jfc:EasyDict, model:nn.Module):
    if optim_jfc.type_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), **(optim_jfc.keywords))
    if optim_jfc.type_name.lower() == 'adam':
        return optim.Adam(model.parameters(), **(optim_jfc.keywords))
    if optim_jfc.type_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), **(optim_jfc.keywords))
    raise NotImplementedError(optim_jfc)
