import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sys
sys.path.append('./src')
from training.training_loop import training_process
import curriculums

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in list(range(torch.cuda.device_count())))

def setup(rank, world_size, port, process_group = "gloo"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    torch.cuda.set_device(rank)
    dist.init_process_group(process_group, rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup(rank, world_size, opt.port, opt.process_group)
    device = torch.device(rank)
    training_process(rank, world_size, opt, device)
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=2000, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, default='CHAIR_128')
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=2000)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--process_group',type=str, default="gloo") # use 'nccl' for high-end GPU device
    parser.add_argument('--load_continue', action='store_true')

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    
    if opt.set_step != None:
        curriculum_step = opt.set_step
    else:
        curriculum_step = 0

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, curriculum_step)
    opt.set_train_step = 0
    
    if opt.load_dir != '':
        opt.set_train_step = opt.set_step
    elif opt.load_continue:
        step_l = sorted(os.listdir(opt.output_dir))
        if len(step_l) > 0:
            step_l = [int(item.replace('_generator.pth','').replace('step','')) for item in step_l[::-1] if '_generator.pth' in item]
            if (len(step_l) > 0) and (max(step_l)>0): 
                opt.set_step = max(step_l)
                opt.set_train_step = opt.set_step
                opt.load_dir = opt.output_dir
        elif 'pretrain_curriculum' in metadata.keys():
            opt.load_dir = opt.output_dir.replace(opt.output_dir.split('/')[-1], metadata['pretrain_curriculum'])
            opt.set_step = metadata['pretrain_model_iter']

    elif 'pretrain_curriculum' in metadata.keys():
        opt.load_dir = opt.output_dir.replace(opt.output_dir.split('/')[-1], metadata['pretrain_curriculum'])
        opt.set_step = metadata['pretrain_model_iter']

    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)