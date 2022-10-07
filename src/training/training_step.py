import math
import numpy as np
import torch

from .utils import *
from .loss import *

def training_step_D(real_imgs, real_positions, generator_ddp, discriminator_ddp, optimizer_D, alpha, scaler, metadata, device):
    with torch.cuda.amp.autocast():
        # Generate images for discriminator training
        with torch.no_grad():
            z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
            z2 = None
            if metadata.get('double_code', False):
                z2 = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist']) 

            split_batch_size = z.shape[0] // metadata['batch_split']
            gen_imgs = []
            gen_positions = []
            for split in range(metadata['batch_split']):

                g_imgs = []
                subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                subset_z2 = z2[split * split_batch_size:(split+1) * split_batch_size] if z2 is not None else None

                output = generator_ddp(subset_z, stage=metadata['img_size'], alpha=alpha, calc_grad=False, z2=subset_z2, region=0, **metadata)
                g_img0, g_pos = output[:2]
                g_imgs.append(g_img0)
                for r in range(1,metadata['num_regions']):
                    output = generator_ddp(subset_z, pitch=g_pos[...,:1], yaw=g_pos[...,1:], stage=metadata['img_size'], alpha=alpha, calc_grad=False, \
                                                z2=subset_z2, region=r, **metadata)
                    g_imgs.append(output[0].reshape(split_batch_size,3,-1))

                g_imgs = torch.cat(g_imgs,-1).reshape(split_batch_size,3,metadata['img_size'],metadata['img_size'])

                gen_imgs.append(g_imgs)
                gen_positions.append(g_pos)

            gen_imgs = torch.cat(gen_imgs, axis=0)
            gen_positions = torch.cat(gen_positions, axis=0)

        real_imgs.requires_grad = True
        if 'Encoder' in metadata['discriminator']:
            r_preds, _, r_pred_position = discriminator_ddp(real_imgs, alpha, **metadata)
        else:
            r_preds = discriminator_ddp(real_imgs, alpha, **metadata)

    if metadata['r1_lambda'] > 0:
        # Gradient penalty
        grad_outputs = torch.ones_like(r_preds)
        grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds), inputs=real_imgs, grad_outputs=grad_outputs, create_graph=True)
        inv_scale = 1./scaler.get_scale()
        grad_real = [p * inv_scale for p in grad_real][0]
    with torch.cuda.amp.autocast():
        if metadata['r1_lambda'] > 0:
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
        else:
            grad_penalty = 0
        
        if 'Encoder' in metadata['discriminator']:
            g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs.detach(), alpha, **metadata)
            latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']    
            position_penalty = torch.mean(1. - torch.cos(g_pred_position - gen_positions)) * metadata['pos_lambda']
            
            if 'real_pose' in metadata and metadata['real_pose'] == True:
                position_penalty += torch.mean(1. - torch.cos(r_pred_position- real_positions)) * metadata['pos_lambda']
    
            identity_penalty = latent_penalty + position_penalty
            d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
        else:
            g_preds = discriminator_ddp(gen_imgs.detach(), alpha, **metadata)
            d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty            

    optimizer_D.zero_grad()
    scaler.scale(d_loss).backward()
    scaler.unscale_(optimizer_D)

    if not ('use_grad_clip'in metadata and metadata['use_grad_clip'] == False): 
        torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])

    scaler.step(optimizer_D)
    return d_loss.item()

def training_step_G(real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, alpha, scaler, metadata, device):
    '''
    Backward the image loss and the point gradient loss together in one iteration
    '''
    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
    z2 = None
    if metadata.get('double_code', False):
        z2 = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist']) 
    
    split_batch_size = z.shape[0] // metadata['batch_split']

    for split in range(metadata['batch_split']):
        with torch.cuda.amp.autocast():
            subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
            subset_z2 = z2[split * split_batch_size:(split+1) * split_batch_size] if z2 is not None else None
            output = generator_ddp(subset_z, stage=metadata['img_size'], alpha=alpha, calc_grad=metadata.get('use_deformnet', False),\
                                   z2=subset_z2, **metadata)

            gen_imgs, gen_positions, transparency, deform_out  = output[:4]

            if 'Encoder' in metadata['discriminator']:
                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                position_penalty = torch.mean(1. - torch.cos(g_pred_position - gen_positions)) * metadata['pos_lambda']
                identity_penalty = latent_penalty + position_penalty
            else:
                g_preds = discriminator_ddp(gen_imgs, alpha, **metadata)
                identity_penalty = 0
            
            #------ constraints
            correct_reg = 0.
            deform_smooth_reg = 0.
            deform_rigid_reg = 0.
            normal_consist_reg = 0.

            if 'correct_reg' in metadata:
                correct_reg += minimal_correction_regularization(deform_out['dsigma']) \
                                * metadata['correct_reg']

            if 'deform_smooth_reg' in metadata:
                deform_smooth_reg += deform_smooth_regularization(deform_out['grad_dx']) \
                                    * metadata['deform_smooth_reg']

            if 'deform_rigid_reg' in metadata:
                deform_rigid_reg += deform_rigid_regularization(deform_out['grad_dx']) \
                                    * metadata['deform_rigid_reg']

            if 'normal_consist_reg' in metadata:
                # NOTE : the density gradient of the source space is detached !!!
                normal_consist_reg += normal_consistent_regularization(deform_out['grad_sigma'].detach(), deform_out['grad_sigma_t']) \
                                    * metadata['normal_consist_reg']

            #------ END constraints

        with torch.cuda.amp.autocast():
            topk_percentage = max(0.99 ** (discriminator_ddp.module.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
            topk_num = math.ceil(topk_percentage * g_preds.shape[0])

            g_preds = torch.topk(g_preds, topk_num, dim=0).values

            g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
            g_loss = g_loss + correct_reg + deform_smooth_reg + deform_rigid_reg + normal_consist_reg

        scaler.scale(g_loss).backward()

    scaler.unscale_(optimizer_G)
    
    if not ('use_grad_clip'in metadata and metadata['use_grad_clip'] == False): 
        torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))

    scaler.step(optimizer_G)
    scaler.update()
    optimizer_G.zero_grad()
    ema.update(generator_ddp.parameters())
    ema2.update(generator_ddp.parameters())

    #-------------------------------
    # regularization of deformation network
    if 'correct_reg' in metadata:
        correct_reg = correct_reg.item()

    if 'deform_smooth_reg' in metadata:
        deform_smooth_reg = deform_smooth_reg.item()

    if 'deform_rigid_reg' in metadata:
        deform_rigid_reg = deform_rigid_reg.item()

    if 'normal_consist_reg' in metadata:
        normal_consist_reg = normal_consist_reg.item()

    return g_loss.item(), topk_num, correct_reg, deform_smooth_reg, deform_rigid_reg, normal_consist_reg


def training_step_G_patchlevel(real_imgs, generator_ddp, ema, ema2, discriminator_ddp, optimizer_G, alpha, scaler, metadata, device):
    '''
    Patch-level forward and backward for saving GPU memory
    '''

    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
    z2 = None
    if metadata.get('double_code', False):
        z2 = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist']) 
    
    split_batch_size = z.shape[0] // metadata['batch_split']

    for split in range(metadata['batch_split']):
        with torch.cuda.amp.autocast():
            subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
            subset_z2 = z2[split * split_batch_size:(split+1) * split_batch_size] if z2 is not None else None

            gen_imgs = []
            with torch.no_grad():
                #---- extract all sampled points, then pass them into the generator
                output = generator_ddp(subset_z, stage=metadata['img_size'], alpha=alpha, \
                                calc_grad=False, return_points=True, \
                                z2=subset_z2, region=0, **metadata)
                
                points_meta_l = []
                points_meta = {}
                points_meta['all_points']  = output[-1]['ret_points']
                points_meta['all_viewdir'] = output[-1]['ret_viewdir']
                points_meta['all_z_vals']  = output[-1]['ret_z_vals']
                points_meta['all_depth']   = output[-1]['ret_depth']
                points_meta_l.append(points_meta)

                gen_imgs0, gen_positions, transparency = output[:3]
                gen_imgs.append(gen_imgs0.reshape(split_batch_size,3,-1))

                for r in range(1,metadata['num_regions']):
                    output = generator_ddp(subset_z, pitch=gen_positions[...,:1],yaw=gen_positions[...,1:],\
                                    stage=metadata['img_size'], alpha=alpha,\
                                    calc_grad=False, z2=subset_z2, region=r, \
                                    return_points=True, **metadata)
                    gen_imgs.append(output[0].reshape(split_batch_size,3,-1))

                    points_meta = {}
                    points_meta['all_points']  = output[-1]['ret_points']
                    points_meta['all_viewdir'] = output[-1]['ret_viewdir']
                    points_meta['all_z_vals']  = output[-1]['ret_z_vals']
                    points_meta['all_depth']   = output[-1]['ret_depth']
                    points_meta_l.append(points_meta)

            gen_imgs = torch.cat(gen_imgs,-1).reshape(split_batch_size,3,metadata['img_size'],metadata['img_size'])
            gen_imgs.requires_grad_(True)

            if 'Encoder' in metadata['discriminator']:
                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                position_penalty = torch.mean(1. - torch.cos(g_pred_position - gen_positions)) * metadata['pos_lambda']
                identity_penalty = latent_penalty + position_penalty
            else:
                g_preds = discriminator_ddp(gen_imgs, alpha, **metadata)
                identity_penalty = 0
                    
        with torch.cuda.amp.autocast():
            topk_percentage = max(0.99 ** (discriminator_ddp.module.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
            topk_num = math.ceil(topk_percentage * g_preds.shape[0])

            g_preds = torch.topk(g_preds, topk_num, dim=0).values
            g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty

        grad_gen_imgs = torch.autograd.grad(outputs=scaler.scale(g_loss), inputs=gen_imgs, create_graph=False)[0] 

        grad_gen_imgs = grad_gen_imgs.reshape(split_batch_size,3,-1)
        grad_gen_imgs = grad_gen_imgs.detach() 

        total_correct_reg = 0.
        total_deform_smooth_reg = 0.
        total_deform_rigid_reg = 0.
        total_normal_consist_reg = 0.
    
        random_ray_mask = None
    
        for r in range(metadata['num_regions']):
            start = r*metadata['img_size']*metadata['img_size']//metadata['num_regions']
            end = start + metadata['img_size']*metadata['img_size']//metadata['num_regions']

            points_meta = points_meta_l[r]
            deform_loss = 0.
            with torch.cuda.amp.autocast():
                subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                subset_z2 = z2[split * split_batch_size:(split+1) * split_batch_size] if z2 is not None else None

                do_calc_grad = metadata.get('use_deformnet', False)
                output = generator_ddp(subset_z, pitch=gen_positions[...,:1],yaw=gen_positions[...,1:],\
                                stage=metadata['img_size'], alpha=alpha,\
                                calc_grad=do_calc_grad, z2=subset_z2,  \
                                region=r, points_meta=points_meta, **metadata)
                
                gen_imgs, _, transparency, deform_out  = output[:4]
                deform_loss = 0
    
            if deform_loss == 0.:
                gen_imgs.backward(grad_gen_imgs[...,start:end])
            else:
                gen_imgs.backward(grad_gen_imgs[...,start:end], retain_graph=True)
                scaler.scale(deform_loss).backward()

        #------------ calculate point loss
        full_points_meta = {}
        for r in range(metadata['num_regions']):
            points_meta = points_meta_l[r]
            for tmp_k, tmp_v in points_meta.items():
                if tmp_k not in full_points_meta:
                    full_points_meta[tmp_k] = []
                full_points_meta[tmp_k].append(tmp_v)

        for tmp_k, tmp_v in full_points_meta.items():
            full_points_meta[tmp_k] = torch.cat(tmp_v, dim=1)

        with torch.cuda.amp.autocast():
            deform_out = generator_ddp(subset_z, pitch=gen_positions[...,:1],yaw=gen_positions[...,1:],\
                                stage=metadata['img_size'], alpha=alpha,\
                                z2=subset_z2,
                                random_ray_mask=random_ray_mask, 
                                points_meta=full_points_meta, 
                                grad_scaler=scaler, 
                                only_calc_grad=True,**metadata)

            correct_reg = 0.
            deform_smooth_reg = 0.
            deform_rigid_reg = 0.
            normal_consist_reg = 0.

            if 'correct_reg' in metadata:
                correct_reg += minimal_correction_regularization(deform_out['dsigma']) \
                                * metadata['correct_reg']

            if 'deform_smooth_reg' in metadata:
                deform_smooth_reg += deform_smooth_regularization(deform_out['grad_dx']) \
                                    * metadata['deform_smooth_reg']

            if 'deform_rigid_reg' in metadata:
                deform_rigid_reg += deform_rigid_regularization(deform_out['grad_dx']) \
                                    * metadata['deform_rigid_reg']

            if 'normal_consist_reg' in metadata:
                # NOTE : the density gradient of the source space is detached !!!
                normal_consist_reg += normal_consistent_regularization(deform_out['grad_sigma'].detach(), deform_out['grad_sigma_t']) \
                                    * metadata['normal_consist_reg']                
                if metadata.get('normal_loss_prog', False):
                    normal_consist_reg = normal_consist_reg * float(metadata['normal_loss_prog_scale'])
            #----------------------- END calculate deformation loss

            deform_loss = correct_reg + deform_smooth_reg + deform_rigid_reg + normal_consist_reg


        if deform_loss != 0.0:
            scaler.scale(deform_loss).backward()

        total_correct_reg        += correct_reg
        total_deform_smooth_reg  += deform_smooth_reg
        total_deform_rigid_reg   += deform_rigid_reg
        total_normal_consist_reg += normal_consist_reg

    scaler.unscale_(optimizer_G)

    if not ('use_grad_clip'in metadata and metadata['use_grad_clip'] == False): 
        torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))

    scaler.step(optimizer_G)
    scaler.update()
    optimizer_G.zero_grad()
    ema.update(generator_ddp.parameters())
    ema2.update(generator_ddp.parameters())

    #-------------------------------
    # regularizations of deformation network
    if 'correct_reg' in metadata:
        total_correct_reg = total_correct_reg.item()

    if 'deform_smooth_reg' in metadata:
        total_deform_smooth_reg = total_deform_smooth_reg.item()

    if 'deform_rigid_reg' in metadata:
        total_deform_rigid_reg = total_deform_rigid_reg.item()

    if 'normal_consist_reg' in metadata:
        total_normal_consist_reg = total_normal_consist_reg.item()


    return g_loss.item(), topk_num, total_correct_reg, total_deform_smooth_reg, total_deform_rigid_reg, total_normal_consist_reg