import os
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image

import datasets
import curriculums
from tqdm import tqdm
import copy

from .loss import *
from .model_setup import *
from .training_step import training_step_D, training_step_G, training_step_G_patchlevel
from .utils import z_sampler ,process_training_status_img

def training_process(rank, world_size, opt, device):
#--------------------------------------------------------------------------------------
# extract training curriculums
    curriculum_step = opt.set_train_step

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, curriculum_step)

#--------------------------------------------------------------------------------------
# set amp gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    if opt.load_dir != '':
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_scaler.pth'%opt.set_step)))
    else:
        if 'scaler_init' in metadata:
            scaler._init_scale = metadata['scaler_init']


    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

#--------------------------------------------------------------------------------------
# set fixed random latent code for visualization purpose
    fixed_z  = z_sampler((16, metadata['latent_dim']), device='cpu', dist=metadata['z_dist'])
    fixed_z2 = None
    if metadata.get('double_code', False):
        fixed_z2 = z_sampler((16, metadata['latent_dim']), device='cpu', dist=metadata['z_dist'])
    
#--------------------------------------------------------------------------------------
# set deformation network
    deformnet_ins = None
    if metadata.get('use_deformnet',False):
        deformnet_ins = set_deformnet(metadata, device, opt)

#--------------------------------------------------------------------------------------
# set generator and discriminator
    generator, ema, ema2 = set_generator(metadata, device, opt, deformnet_ins=deformnet_ins)
    discriminator = set_discriminator(metadata,device,opt)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)

    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

#--------------------------------------------------------------------------------------
# set optimizers
    optimizer_G = set_optimizer_G(generator_ddp, metadata, opt)
    optimizer_D = set_optimizer_D(discriminator_ddp, metadata, opt)

    torch.cuda.empty_cache()

    generator_losses = []
    discriminator_losses = []

    generator.step = opt.set_train_step
    discriminator.step = opt.set_train_step

    generator.set_device(device)
    # ----------
    #  Training
    # ----------
    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True) # Keeps track of total training
    total_progress_bar.update(discriminator.epoch) # Keeps track of progress to next stage
    interior_step_bar = tqdm(dynamic_ncols=True)

#--------------------------------------------------------------------------------------
# get dataset
    metadata = curriculums.extract_metadata(curriculum, discriminator.step)
    dataset = getattr(datasets, metadata['dataset'])(**metadata)

#--------------------------------------------------------------------------------------
# main training loop
    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step) # different steps may have different settings due to progressive growing strategy
        
        # Set learning rates
        set_learning_rate_G(optimizer_G, metadata)
        set_learning_rate_D(optimizer_D, metadata)
        
        # if current batchsize not equal to metadata, meaning that the progressive growing moves to the next stage
        if not dataloader or dataset.img_size != metadata['img_size']: 
            dataset = getattr(datasets, metadata['dataset'])(**metadata)
            dataset.img_size = metadata['img_size']
            dataset.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((metadata['img_size'], metadata['img_size']), interpolation=2)])
            dataloader, CHANNELS = datasets.get_dataset_distributed(dataset,
                            world_size,
                            rank,
                            **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step) # next step to grow up the resolution
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step) # previous step for resolution grown

            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        #--------------------------------------------------------------------------------------
        # trainging iterations
        for i, (imgs, poses) in enumerate(dataloader):
            # save model
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                torch.save(ema, os.path.join(opt.output_dir, 'step%06d_ema.pth'%discriminator.step))
                torch.save(ema2, os.path.join(opt.output_dir, 'step%06d_ema2.pth'%discriminator.step))
                torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'step%06d_generator.pth'%discriminator.step))
                torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'step%06d_discriminator.pth'%discriminator.step))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'step%06d_optimizer_G.pth'%discriminator.step))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'step%06d_optimizer_D.pth'%discriminator.step))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'step%06d_scaler.pth'%discriminator.step))
                torch.cuda.empty_cache()
                
            dist.barrier()

            metadata = curriculums.extract_metadata(curriculum, discriminator.step) # extract training settings for current iteration
            # if dataloader.batch_size != metadata['batch_size']: break # move to next stage
            if dataset.img_size != metadata['img_size']: break # move to next stage

            if scaler.get_scale() < 1:
                scaler.update(1.)

            if metadata['fade_steps'] == 0:
                alpha = 1
            elif discriminator.step < metadata['fade_steps']:
                alpha = 1
            else:
                alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))  # fading parameter for progressive growing

            real_imgs = imgs.to(device, non_blocking=True)
            if 'real_pose' in metadata and metadata['real_pose'] == True:
                real_poses = poses.to(device, non_blocking=True)
            else:
                real_poses = poses

            metadata['nerf_noise'] = 0. #max(0, 1. - discriminator.step/5000.)

            generator_ddp.train()
            discriminator_ddp.train()
            
            #--------------------------------------------------------------------------------------
            # TRAIN DISCRIMINATOR
            d_loss = training_step_D(real_imgs, real_poses, generator_ddp, discriminator_ddp, optimizer_D, alpha, scaler, metadata, device)
            discriminator_losses.append(d_loss)

            # TRAIN GENERATOR
            training_step_G_func = training_step_G_patchlevel if (metadata.get('num_regions', 1) > 1) else training_step_G
            g_loss, topk_num, correct_reg, deform_smooth_reg, deform_rigid_reg, normal_consist_reg, \
                                = training_step_G_func(real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, alpha, scaler, metadata, device)
                
            generator_losses.append(g_loss)

            #--------------------------------------------------------------------------------------
            # print and save
            if rank == 0:
                if i%opt.print_interval == 0:
                    interior_step_bar.update(opt.print_interval)
                    tqdm.write(f"[Experiment: {opt.output_dir}] [GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] [Epoch: {discriminator.epoch}/{opt.n_epochs}] [D loss: {d_loss:.2e}] [G loss: {g_loss:.2e}] [correct_reg: {correct_reg:.2e}] [deform_smooth: {deform_smooth_reg:.2e}] [deform_rigid: {deform_rigid_reg:.2e}] [normal_consist: {normal_consist_reg:.2e}] [Step: {discriminator.step}] [Alpha: {alpha:.2f}] [Img Size: {metadata['img_size']}] [Batch Size: {metadata['batch_size']}] [TopK: {topk_num}] [Scale: {scaler.get_scale()}]")

                # save fixed angle generated images
                if discriminator.step % opt.sample_interval == 0:
                    os.makedirs(os.path.join(opt.output_dir, 'training_status'), exist_ok=True)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            img_size  = copied_metadata['img_size']
                            max_depth = copied_metadata['ray_end']
                            inp_z2 = fixed_z2.to(device) if fixed_z2 is not None else None
                            full_output = generator_ddp.module.staged_forward(fixed_z.to(device), stage=metadata['img_size'], alpha=alpha, calc_grad=False, z2=inp_z2, randomize=not metadata.get('sample_no_random', False), **copied_metadata)
                            full_imgs, full_depth = full_output[0], full_output[1]

                            noCorrect_output = generator_ddp.module.staged_forward(fixed_z.to(device), stage=metadata['img_size'], alpha=alpha, calc_grad=False, z2=inp_z2, no_correct=True, randomize=not metadata.get('sample_no_random', False), **copied_metadata)
                            noCor_imgs, noCor_depth = noCorrect_output[0], noCorrect_output[1]
                            
                            template_output  = generator_ddp.module.staged_forward(fixed_z.to(device), stage=metadata['img_size'], alpha=alpha, calc_grad=False, z2=inp_z2, no_correct=True, no_deformation=True, randomize=not metadata.get('sample_no_random', False), **copied_metadata)
                            temp_imgs,  temp_depth  = template_output[0],  template_output[1]
                        
                            final_img = process_training_status_img(full_imgs, full_depth, noCor_imgs, noCor_depth, temp_imgs, temp_depth, img_size, max_depth)
                            im = Image.fromarray(final_img)
                            im.save(os.path.join(opt.output_dir, "training_status/%06d_fixed.png"%discriminator.step))
                        
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            inp_z2 = fixed_z2.to(device) if fixed_z2 is not None else None
                            full_output = generator_ddp.module.staged_forward(fixed_z.to(device), stage=metadata['img_size'], alpha=alpha, calc_grad=False, z2=inp_z2, randomize=not metadata.get('sample_no_random', False), **copied_metadata)
                            full_imgs, full_depth = full_output[0], full_output[1]

                            noCorrect_output = generator_ddp.module.staged_forward(fixed_z.to(device), stage=metadata['img_size'], alpha=alpha, calc_grad=False, z2=inp_z2, no_correct=True, randomize=not metadata.get('sample_no_random', False), **copied_metadata)
                            noCor_imgs, noCor_depth = noCorrect_output[0], noCorrect_output[1]
                            
                            template_output  = generator_ddp.module.staged_forward(fixed_z.to(device), stage=metadata['img_size'], alpha=alpha, calc_grad=False, z2=inp_z2, no_correct=True, no_deformation=True, randomize=not metadata.get('sample_no_random', False), **copied_metadata)
                            temp_imgs,  temp_depth  = template_output[0],  template_output[1]
                        
                            final_img = process_training_status_img(full_imgs, full_depth, noCor_imgs, noCor_depth, temp_imgs, temp_depth, img_size, max_depth)
                            im = Image.fromarray(final_img)
                            im.save(os.path.join(opt.output_dir, "training_status/%06d_tilted.png"%discriminator.step))

                # save_model
                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))
                
                torch.cuda.empty_cache()
            
            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1