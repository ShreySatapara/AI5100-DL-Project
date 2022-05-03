
import importlib
import fire
import os
import copy
import torch

from _path_init import *
from objectDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from objectDet3D.utils.utils import jfc_from_file

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(config:str="config/config.py",
        gpu:int=0, 
        checkpoint_path:str="retinanet_79.pth",
        split_to_test:str='validation'):
    # Read Config
    jfc = jfc_from_file(config)
    
    # Force GPU selection in command line
    jfc.trainer.gpu = gpu
    torch.cuda.set_device(jfc.trainer.gpu)
    
    # Set up dataset and dataloader
    is_test_train = split_to_test == 'training'
    if split_to_test == 'training':
        dataset_name = jfc.data.train_dataset
    elif split_to_test == 'test':
        dataset_name = jfc.data.test_dataset
        jfc.is_running_test_set = True
    else:
        dataset_name = jfc.data.val_dataset
    dataset = DATASET_DICT[dataset_name](jfc, split_to_test)

    # Create the model
    detector = DETECTOR_DICT[jfc.detector.name](jfc.detector)
    detector = detector.cuda()

    state_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(jfc.trainer.gpu))
    new_dict = state_dict.copy()
    detector.load_state_dict(new_dict, strict=False)
    detector.eval()

    if 'eval_function' in jfc.trainer:
        evaluate_detection = PIPELINE_DICT[jfc.trainer.eval_function]
        print("Found evaluate function")
    else:
        raise KeyError("evluate_func not found in Config")

    # Run evaluation
    evaluate_detection(jfc, detector, dataset, None, 0, result_path_split=split_to_test)
    print('finish')
if __name__ == '__main__':
    fire.Fire(main)
