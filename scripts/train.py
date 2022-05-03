
import numpy as np

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
import os
from torch.utils.tensorboard import SummaryWriter
import coloredlogs
from tqdm import tqdm
from fire import Fire

import objectDet3D.data.kitti.dataset
from objectDet3D.utils.timer import Timer
from objectDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from objectDet3D.utils.utils import LossLogger, jfc_from_file
from objectDet3D.networks.optimizers import optimizers, schedulers
from objectDet3D.networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from objectDet3D.evaluator.kitti.evaluate import evaluate
from _path_init import *
import pprint


os.system("python3 depth_precompute.py")
os.system("python3 disparity_compute.py")
os.system("python3 imdb_precompute_3d.py")
os.system("python3 imdb_precompute_test.py")

def main(config="config/config.py"):
   

   
    jfc = jfc_from_file(config)

    
    jfc.dist = EasyDict()
    jfc.dist.world_size = 1
    jfc.dist.local_rank = 1
    dist = False
    logng     = True
    evaluate  = True

   
    rcdir = os.path.join(jfc.path.log_path, "default" + f"config={config}")
    
    if logng: 
        if os.path.isdir(rcdir):
            os.system("rm -r {}".format(rcdir))
        writer = SummaryWriter(rcdir)
       
        formatted_jfc = pprint.pformat(jfc)
        writer.add_text("config.py", formatted_jfc.replace(' ', '&nbsp;').replace('\n', '  \n'))
    else:
        writer = None

    
    gpu = min(jfc.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(jfc.trainer, 'cudnn', False)
    torch.cuda.set_device(gpu)
    
    training_dataset = DATASET_DICT[jfc.data.training_dataset](jfc)
    validation_dataset = DATASET_DICT[jfc.data.val_dataset](jfc, "validation")

    train_dataloader = DataLoader(training_dataset, num_workers=jfc.data.num_workers,
                                  batch_size=jfc.data.batch_size, collate_fn=training_dataset.collate_fn, shuffle=True, drop_last=True,
                                  sampler=torch.utils.data.DistributedSampler(training_dataset, num_replicas=1, rank=1, shuffle=True))

    
    dtctr = DETECTOR_DICT[jfc.detector.name](jfc.detector)

   
    old_checkpoint = getattr(jfc.path, 'pretrained_checkpoint', None)
    if old_checkpoint is not None:
        state_dict = torch.load(old_checkpoint, map_location='cpu')
        dtctr.load_state_dict(state_dict)

   
    dtctr = dtctr.cuda()
    dtctr.train()

    
    if logng:
        str1 = dtctr.__str__().replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text("Model architecture", str1)
        num_parameters = get_num_parameters(dtctr)
        print(f'No. of parameters used in trainig: {num_parameters}')
    
    
    optim = optimizers.build_optimizer(jfc.optimizer, dtctr)

   
    scheduler_cfg = getattr(jfc, 'scheduler', None)
    schdlr = schedulers.build_scheduler(scheduler_cfg, optim)
    iter_based = getattr(scheduler_cfg, "iter_based", False)

    
    training_loss_logger =  LossLogger(writer, 'train') if logng else None

    
    if 'train_function' in jfc.trainer:
        training_dection = PIPELINE_DICT[jfc.trainer.train_function]
   
    if 'eval_function' in jfc.trainer:
        evaluate_detection = PIPELINE_DICT[jfc.trainer.eval_function]
    else:
        evaluate_detection = None
  
    timer = Timer()

    print('Num training images: {}'.format(len(training_dataset)))

    step_global = 0

    for num_epochs in range(jfc.trainer.max_epochs):
       
        dtctr.train()
        if training_loss_logger:
            training_loss_logger.reset()
        for iter_num, data in enumerate(train_dataloader):
            training_dection(data, dtctr, optim, writer, training_loss_logger, step_global, num_epochs, jfc)

            step_global += 1

            if iter_based:
                schdlr.step()

            if logng and step_global % jfc.trainer.dp_iteration == 0:
                
                if 'total_loss' not in training_loss_logger.loss_stats:
                    print(f"\nIn epoch {num_epochs}, iteration:{iter_num}, step_global:{step_global}, total_loss not found in logger.")
                else:
                    log_str = 'Epoch: {} --- Iteration: {}  --- Running loss: {:1.5f} --- eta:{}'.format(
                        num_epochs, iter_num, training_loss_logger.loss_stats['total_loss'].avg,
                        timer.compute_eta(step_global, len(train_dataloader) * jfc.trainer.max_epochs))
                    print(log_str, end='\r')
                    writer.add_text("training_log/train", log_str, step_global)
                    training_loss_logger.log(step_global)

        if not iter_based:
            schdlr.step()

       
        if logng:
            torch.save(dtctr.module.state_dict() if dist else dtctr.state_dict(), os.path.join(
                jfc.path.checkpoint_path, '{}_latest.pth'.format(
                    jfc.dtctr.name)
                )
            )
        if logng and (num_epochs + 1) % jfc.trainer.sv_iteration == 0:
            torch.save(dtctr.module.state_dict() if dist else dtctr.state_dict(), os.path.join(
                jfc.path.checkpoint_path, '{}_{}.pth'.format(
                    jfc.dtctr.name,num_epochs)
                )
            )

        
        if evaluate and evaluate_detection is not None and jfc.trainer.tt_iteration > 0 and (num_epochs + 1) % jfc.trainer.tt_iteration == 0:
            print("\n/Testing at epoch {} ".format(num_epochs))
            evaluate_detection(jfc, dtctr.module if dist else dtctr, validation_dataset, writer, num_epochs)


        if dist:
            torch.distributed.barrier() 

        if logng:
            writer.flush()

if __name__ == '__main__':
    Fire(main)
