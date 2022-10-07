"""Implicit generator for 3D volumes"""

import random
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from .volumetric_rendering import *

class ImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, deformnet_ins = None, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None,**kwargs)
        self.epoch = 0
        self.step = 0
        kernel = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16
        self.GaussianBlurKernel = torch.zeros(3, 3, 3, 3)
        self.GaussianBlurKernel[0,0] = kernel
        self.GaussianBlurKernel[1,1] = kernel
        self.GaussianBlurKernel[2,2] = kernel

        self.deformnet = deformnet_ins

    def set_device(self, device):
        self.device = device
        self.siren.device = device
        if self.deformnet is not None:
            self.deformnet.device = device

    def calc_density_gradient(self, x, sigma, **kwargs):
        '''
        Input: 
            x :    (B, N, 3) or (B, R, M, 3)   3D points
            sigma: (B, N, 1) or (B, R, M, 1)  volume density
        Ouput: 
            grad_sigma : (B, N, 3) or (B, R, M, 3) the gradient of density
        '''
        grad_outputs = torch.ones_like(sigma)
        grad_sigma = torch.autograd.grad(sigma, [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=False)[0]
        normal_norm = torch.norm(grad_sigma, dim=-1)
        return grad_sigma


    def generate_avg_frequencies(self):
        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)

    def generate_avg_frequencies_deform(self):
        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.deformnet.mapping_network(z)
        self.avg_freq_deform = frequencies.mean(0, keepdim=True)
        self.avg_shifts_deform = phase_shifts.mean(0, keepdim=True)
    

    def get_frequencies(self, z, z2=None, psi=1.0):
        if z2 is None:
            frequencies, phase_shifts = self.siren.mapping_network(z)
        else:
            frequencies, phase_shifts = self.siren.mapping_network(z2)

        freq_deform, shifts_deform = None, None
        if self.deformnet is not None:
            freq_deform, shifts_deform = self.deformnet.mapping_network(z)
        
        
        if psi != 1.0:
            freq_deform   = self.avg_freq_deform   + psi * (freq_deform   - self.avg_freq_deform)
            shifts_deform = self.avg_shifts_deform + psi * (shifts_deform - self.avg_shifts_deform)
            frequencies   = self.avg_frequencies   + psi * (frequencies   - self.avg_frequencies)
            phase_shifts  = self.avg_phase_shifts  + psi * (phase_shifts  - self.avg_phase_shifts)

        return freq_deform, shifts_deform, frequencies, phase_shifts

    def deform_input_points(self, z, points, psi=0.7, max_batch_size=64**3):
        assert self.deformnet is not None
        if len(points.shape) == 2:
            points = points.view(1,-1,3)

        batch_size = points.shape[0]
        if not psi == 1:
            self.generate_avg_frequencies_deform()

        with torch.no_grad():
            raw_freq_deform, raw_shifts_deform = self.deformnet.mapping_network(z)
            if not psi == 1:
                trunc_freq_deform   = self.avg_freq_deform + psi * (raw_freq_deform - self.avg_freq_deform)
                trunc_shifts_deform = self.avg_shifts_deform + psi * (raw_shifts_deform - self.avg_shifts_deform)
            else:
                trunc_freq_deform   = raw_freq_deform
                trunc_shifts_deform = raw_shifts_deform

            dx = torch.zeros_like(points, device=self.device)
            dsigma = torch.zeros((*points.shape[:2], 1), device=self.device)

            for b in range(batch_size):
                head = 0
                while head < points.shape[1]:
                    tail = head + max_batch_size
                    dx[b:b+1, head:tail], dsigma[b:b+1, head:tail] = self.deformnet.forward_with_frequencies_phase_shifts(points[b:b+1, head:tail], trunc_freq_deform[b:b+1], trunc_shifts_deform[b:b+1])
                    head += max_batch_size
        return dx, dsigma


    def forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                hierarchical_sample, stage, alpha, sample_dist=None, lock_view_dependence=False, calc_grad=False, z2=None, z_bg=None, \
                pitch=None, yaw=None, region=None, return_points=False, points_meta=None, only_sample_ray=False, \
                only_calc_grad=False, **kwargs):
        
        if only_calc_grad:
            ''' Only calculate the point gradient near the depth
            '''
            return self.forward_calc_grad(z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                        hierarchical_sample, stage, alpha, sample_dist=sample_dist, lock_view_dependence=lock_view_dependence, calc_grad=calc_grad, z2=z2, z_bg=z_bg, \
                        pitch=pitch, yaw=yaw, region=region, return_points=return_points, points_meta=points_meta, **kwargs)
        
        elif points_meta is not None:
            ''' Forward given the sampled points along the ray
            '''
            return self.forward_sorted_ray(z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                        hierarchical_sample, stage, alpha, sample_dist=sample_dist, lock_view_dependence=lock_view_dependence, calc_grad=calc_grad, z2=z2, z_bg=z_bg, \
                        pitch=pitch, yaw=yaw, region=region, return_points=return_points, points_meta=points_meta, **kwargs)
        
        else:
            ''' First sample camera, then get the network output
            '''
            return self.forward_basic(z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                        hierarchical_sample, stage, alpha, sample_dist=sample_dist, lock_view_dependence=lock_view_dependence, calc_grad=calc_grad, z2=z2, z_bg=z_bg, \
                        pitch=pitch, yaw=yaw, region=region, return_points=return_points, only_sample_ray=only_sample_ray, **kwargs)

    def calc_point_gradient_near_depth(self, freq_deform, shifts_deform, frequencies, phase_shifts, points_meta, **kwargs):
        all_points  = points_meta['all_points']
        all_viewdir = points_meta['all_viewdir']
        all_z_vals  = points_meta['all_z_vals']
        depth       = points_meta['all_depth']

        batch_size, total_rays, total_num_steps, _ = all_points.shape
        all_points  = all_points.reshape(batch_size, total_rays*total_num_steps, 3)
        all_viewdir = all_viewdir.reshape(batch_size, total_rays*total_num_steps, 3)

        #-------------------- calculate random ray samples and number of points near the depth
        reg_ray_sample_ratio = float(kwargs['reg_ray_sample_ratio'])
        num_reg_sample = round(reg_ray_sample_ratio * total_rays)
        reg_near_depth_ratio = float(kwargs['reg_near_depth_ratio'])
        num_near_depth = round(reg_near_depth_ratio * total_num_steps)

        #-------------------- sample rays given the sample ratio
        random_ray_indices = torch.randint(total_rays, (batch_size, num_reg_sample, 1, 1), device = all_points.device)
        sel_z_vals = torch.gather(all_z_vals, 1, random_ray_indices.expand(-1, -1, total_num_steps, 1)) 
        sel_depth = torch.gather(depth.unsqueeze(-1), 1, random_ray_indices)  
        sel_points = torch.gather(all_points.reshape(batch_size, -1, total_num_steps, 3),  1, random_ray_indices.expand(-1, -1, total_num_steps, 3))
        sel_viewdir = torch.gather(all_viewdir.reshape(batch_size, -1, total_num_steps, 3), 1, random_ray_indices.expand(-1, -1, total_num_steps, 3))

        sel_z_vals = sel_z_vals.reshape(-1, total_num_steps, 1)
        sel_depth  = sel_depth.reshape(-1, 1, 1)
        sel_points = sel_points.reshape(-1, total_num_steps, 3)
        sel_viewdir = sel_viewdir.reshape(-1, total_num_steps, 3)

        #-------------------- select points near depth
        distance_to_depth = torch.abs(sel_z_vals - sel_depth)
        _, near_indices = torch.topk(distance_to_depth, num_near_depth, dim=-2, largest=False)
        near_depth_mask = torch.zeros(sel_z_vals.shape, dtype=torch.bool, device = sel_z_vals.device) 
        near_depth_mask = near_depth_mask.scatter_(-2, near_indices, True).squeeze(-1)

        near_points = sel_points[near_depth_mask].view(batch_size, -1, 3)
        near_points.requires_grad_(True)
        near_viewdir = sel_viewdir[near_depth_mask].view(batch_size, -1, 3)

        rest_points = sel_points[~near_depth_mask].view(batch_size, -1, 3)
        rest_points.requires_grad_(True)
        rest_viewdir = sel_viewdir[~near_depth_mask].view(batch_size, -1, 3)

        #-------------------- calculate displacement gradient 
        near_points.requires_grad_(True)
        near_deform_out = self.deformnet.forward_with_frequencies_phase_shifts(near_points, freq_deform, shifts_deform)
        near_dx, near_dsigma = near_deform_out[0], near_deform_out[1]
        near_grad_dx = self.deformnet.calc_disp_gradient(near_points, near_dx, **kwargs)

        rest_deform_out = self.deformnet.forward_with_frequencies_phase_shifts(rest_points, freq_deform, shifts_deform)
        rest_dx, rest_dsigma = rest_deform_out[0], rest_deform_out[1]
        rest_grad_dx = self.deformnet.calc_disp_gradient(rest_points, rest_dx, **kwargs)
        
        grad_dx = torch.cat([near_grad_dx, rest_grad_dx], dim=1)

        #------------------- calculate density gradient
        near_points_t = near_points + near_dx
        near_output_t = self.siren.forward_with_frequencies_phase_shifts(near_points_t, frequencies, phase_shifts, \
                                    ray_directions=near_viewdir)
        
        near_sigma_t = near_output_t[...,3:4]
        near_sigma = near_sigma_t + near_dsigma

        grad_sigma_t = self.calc_density_gradient(near_points_t, near_sigma_t, **kwargs)
        grad_sigma = self.calc_density_gradient(near_points, near_sigma, **kwargs)

        dsigma_cat = torch.cat([near_dsigma, rest_dsigma], dim=1)
        dx_cat = torch.cat([near_dx, rest_dx], dim=1)

        deform_out = {}
        deform_out['dx_all'] = dx_cat   
        deform_out['dx'] = dx_cat
        deform_out['dsigma']=  dsigma_cat
        deform_out['grad_dx'] = grad_dx
        deform_out['grad_sigma'], deform_out['grad_sigma_t'] = grad_sigma, grad_sigma_t

        return deform_out

    def forward_calc_grad(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                hierarchical_sample, stage, alpha, sample_dist=None, lock_view_dependence=False, calc_grad=False, z2=None, z_bg=None, \
                pitch=None, yaw=None, region=None, return_points=False, points_meta=None, **kwargs):
        
        batch_size = z.shape[0]
        if not 'delta_final' in kwargs:
            kwargs['delta_final'] = 1e10
        #------------ mapping network ------------
        freq_deform, shifts_deform, frequencies, phase_shifts = self.get_frequencies(z, z2=z2)
        deform_out = self.calc_point_gradient_near_depth(freq_deform, shifts_deform, frequencies, phase_shifts, points_meta, **kwargs)
        return deform_out

    def forward_sorted_ray(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                hierarchical_sample, stage, alpha, sample_dist=None, lock_view_dependence=False, calc_grad=False, z2=None, z_bg=None, \
                pitch=None, yaw=None, region=None, return_points=False, points_meta=None, **kwargs):

        #------------!!!!!!!!!!!!!!! assume the input points have been sorted by z_vals
        batch_size = z.shape[0]
        if not 'delta_final' in kwargs:
            kwargs['delta_final'] = 1e10

        kwargs['mean_sample_delta'] = (ray_end - ray_start) / (num_steps * 2)

        dx, dsigma = None, None # the points offset and the density correction        
        dsigma_cat = None
        deform_out = {}

        #------------ mapping network ------------
        freq_deform, shifts_deform, frequencies, phase_shifts = self.get_frequencies(z, z2=z2)

        #------------  coarse sample ------------
        all_points  = points_meta['all_points']
        all_viewdir = points_meta['all_viewdir']
        all_z_vals  = points_meta['all_z_vals']

        return self.forward_with_freq_points(freq_deform, shifts_deform, frequencies, phase_shifts, \
                                all_points, all_viewdir, all_z_vals, pitch=pitch, yaw=yaw, stage=stage, alpha=alpha, **kwargs)

    def forward_with_freq_points(self, freq_deform, shifts_deform, frequencies, phase_shifts, \
                                all_points, all_viewdir, all_z_vals, stage=128, alpha=1.0, pitch=None, yaw=None, **kwargs):
        batch_size, total_rays, total_num_steps, _ = all_points.shape
        all_points  = all_points.reshape(batch_size, total_rays*total_num_steps, 3)
        all_viewdir = all_viewdir.reshape(batch_size, total_rays*total_num_steps, 3)

        if self.deformnet is not None:
            dx, dsigma = self.deformnet.forward_with_frequencies_phase_shifts(all_points, freq_deform, shifts_deform)[:2]            
            all_points_t = all_points + dx
        else:
            all_points_t = all_points

        all_outputs = self.siren.forward_with_frequencies_phase_shifts(all_points_t, frequencies, phase_shifts, ray_directions=all_viewdir, stage=stage, alpha=alpha)
        all_outputs = all_outputs.reshape(batch_size, total_rays, total_num_steps, 4)

        if self.deformnet is not None:
            rgb, sigma_t = all_outputs[...,:3], all_outputs[...,3:4]
            sigma = sigma_t + dsigma.view(batch_size, total_rays, total_num_steps, 1)
            all_outputs = torch.cat([rgb, sigma], dim = -1)

        pixels, depth, weights, T = fancy_integration(all_outputs, all_z_vals, device=self.device, \
                            white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], \
                            rgb_clamp_mode=kwargs['rgb_clamp_mode'], noise_std=kwargs['nerf_noise'], delta_final=kwargs['delta_final'],\
                            force_lastPoint_white = kwargs.get('force_lastPoint_white', False))

        pixels = pixels.reshape((batch_size, -1, 3))
        pixels = pixels.permute(0, 2, 1).contiguous() * 2 - 1

        if 'reg_ray_sample_ratio' in kwargs.keys():
            reg_ray_sample_ratio = float(kwargs['reg_ray_sample_ratio'])
            num_reg_sample = round(reg_ray_sample_ratio * total_rays)
        
        dsigma = dsigma.reshape(batch_size, total_rays, total_num_steps, 1)
        random_ray_indices = torch.randint(total_rays, (batch_size, num_reg_sample, 1, 1), device = dsigma.device)
        dsigma_cat = torch.gather(dsigma, 1, random_ray_indices.expand(-1, -1, total_num_steps, 1)) 
        deform_out = {}
        deform_out['dsigma'] = dsigma_cat

        pose_info = None
        if pitch is not None:
            pose_info = torch.cat([pitch, yaw], -1)

        return pixels, pose_info, T, deform_out

    def forward_basic(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                hierarchical_sample, stage, alpha, sample_dist=None, lock_view_dependence=False, calc_grad=False, z2=None, \
                pitch=None, yaw=None, region=None, return_points=False, **kwargs):

        batch_size = z.shape[0]
        if not 'delta_final' in kwargs:
            kwargs['delta_final'] = 1e10

        dx, dsigma = None, None # the points offset and the density correction        
        dx_fine, dsigma_fine = None, None
        deform_out = {}

        #------------ mapping network ------------
        freq_deform, shifts_deform, frequencies, phase_shifts = self.get_frequencies(z, z2=z2)
        #------------  coarse sample ------------

        with torch.no_grad():
            uniform_num_steps = num_steps
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, uniform_num_steps, resolution=(img_size, img_size), \
                                                device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            if region is not None:
                start = region*points_cam.shape[1]//kwargs['num_regions']
                end = start + points_cam.shape[1]//kwargs['num_regions']
                
                points_cam = points_cam[:,start:end]
                z_vals = z_vals[:,start:end]
                rays_d_cam = rays_d_cam[:,start:end]            
            
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, \
                                                                                    pitch=pitch, yaw=yaw,
                                                                                    h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, \
                                                                                    randomize = not kwargs.get('sample_no_random', False))
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, uniform_num_steps, -1)
            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            total_rays = img_size*img_size//kwargs['num_regions'] if region is not None else img_size*img_size

            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, total_rays*uniform_num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, total_rays*uniform_num_steps, 3)

        if self.deformnet is not None:
            dx, dsigma = self.deformnet.forward_with_frequencies_phase_shifts(transformed_points, freq_deform, shifts_deform)[:2]
            transformed_points_t = transformed_points + dx
        else:
            transformed_points_t = transformed_points

        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points_t, frequencies, phase_shifts, \
                                    ray_directions=transformed_ray_directions_expanded, stage=stage, alpha=alpha)
        coarse_output = coarse_output.reshape(batch_size, total_rays, uniform_num_steps, 4)

        if self.deformnet is not None:
            rgb, sigma_t = coarse_output[...,:3], coarse_output[...,3:4]
            sigma = sigma_t + dsigma.view(batch_size, total_rays, uniform_num_steps, 1) 
            coarse_output = torch.cat([rgb, sigma], dim = -1)

        if hierarchical_sample:
            with torch.no_grad():
                _, _, weights, _ = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], \
                            rgb_clamp_mode=kwargs['rgb_clamp_mode'], noise_std=kwargs['nerf_noise'], \
                            white_back = kwargs.get('white_back',False), delta_final=kwargs['delta_final'])
                            
                weights = weights.reshape(batch_size * total_rays, num_steps) + 1e-5
                z_vals = z_vals.reshape(batch_size * total_rays, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, total_rays, num_steps, 1)

                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                    num_steps, det=kwargs.get('sample_no_random', False)).detach() # batch_size, num_pixels**2, num_steps

                fine_z_vals = fine_z_vals.reshape(batch_size, total_rays, num_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, total_rays*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            if self.deformnet is not None:
                dx_fine, dsigma_fine = self.deformnet.forward_with_frequencies_phase_shifts(fine_points, freq_deform, shifts_deform)[:2]
                fine_points_t = fine_points + dx_fine
            else:
                fine_points_t = fine_points

            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points_t, frequencies, phase_shifts, \
                                        ray_directions=transformed_ray_directions_expanded, stage=stage, alpha=alpha).reshape(batch_size, total_rays, -1, 4)

            if self.deformnet is not None:
                rgb_fine, sigma_fine_t = fine_output[...,:3], fine_output[...,3:4]
                sigma_fine = sigma_fine_t + dsigma_fine.view(batch_size, total_rays, num_steps, 1)
                fine_output = torch.cat([rgb_fine, sigma_fine], dim = -1)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals  = torch.cat([fine_z_vals, z_vals], dim = -2)
        
            _, indices  = torch.sort(all_z_vals, dim=-2)
            all_z_vals  = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))

        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        pixels, depth, weights, T = fancy_integration(all_outputs, all_z_vals, device=self.device, \
                            white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], \
                            rgb_clamp_mode=kwargs['rgb_clamp_mode'], noise_std=kwargs['nerf_noise'], delta_final=kwargs['delta_final'],\
                            force_lastPoint_white = kwargs.get('force_lastPoint_white', False))

        if region is None:
            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        else:
            pixels = pixels.reshape((batch_size, -1, 3))
            pixels = pixels.permute(0, 2, 1).contiguous() * 2 - 1

        if return_points or calc_grad:
            ret_z_vals  = all_z_vals

            ret_points  = torch.cat([fine_points.view(batch_size, total_rays, num_steps, 3), \
                                    transformed_points.view(batch_size, total_rays, num_steps, 3)], dim=-2)
            ret_viewdir = transformed_ray_directions_expanded.view(batch_size, total_rays, num_steps, 3)
            ret_viewdir = torch.cat([ret_viewdir, ret_viewdir], dim=-2)
            
            ret_points  = torch.gather(ret_points, -2, indices.expand(-1,-1,-1,3))
            ret_viewdir = torch.gather(ret_viewdir, -2, indices.expand(-1,-1,-1,3))

            if return_points:
                deform_out['ret_z_vals']  = ret_z_vals
                deform_out['ret_points']  = ret_points
                deform_out['ret_viewdir'] = ret_viewdir
                deform_out['ret_depth']   = depth.reshape(batch_size, total_rays, 1)

            if calc_grad:
                points_meta = {}
                points_meta['all_points']  = ret_points
                points_meta['all_viewdir'] = ret_viewdir
                points_meta['all_z_vals']  = ret_z_vals
                points_meta['all_depth']   = depth.reshape(batch_size, total_rays, 1)
                gradient_out = self.calc_point_gradient_near_depth(freq_deform, shifts_deform, frequencies, phase_shifts, points_meta, **kwargs)
                for k, v in gradient_out.items():
                    deform_out[k] = v

        return pixels, torch.cat([pitch, yaw], -1), T, deform_out

    def staged_forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, stage, alpha, \
                            z2=None, psi=0.7, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, \
                            far_clip=2, sample_dist=None, hierarchical_sample=False,randomize=True,sigma_only=False,rgb_only=-1, \
                            debug=False, no_correct=False, no_deformation=False, render_normal=False, z_bg=None, **kwargs):
        batch_size = z.shape[0]
        if not 'delta_final' in kwargs:
            kwargs['delta_final'] = 1e10
        deform_out = {}

        with torch.no_grad():
            raw_freq_deform, raw_shifts_deform, raw_frequencies, raw_phase_shifts = self.get_frequencies(z, z2=z2)

            if not psi == 1:
                self.generate_avg_frequencies()
                if self.deformnet is not None:
                    self.generate_avg_frequencies_deform()
                truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
                truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)
                trunc_freq_deform   = self.avg_freq_deform + psi * (raw_freq_deform - self.avg_freq_deform)
                trunc_shifts_deform = self.avg_shifts_deform + psi * (raw_shifts_deform - self.avg_shifts_deform)
            else:
                truncated_frequencies, truncated_phase_shifts = raw_frequencies, raw_phase_shifts
                trunc_freq_deform, trunc_shifts_deform        = raw_freq_deform, raw_shifts_deform
            
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=randomize)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
                if kwargs.get('inverse_lock_view_dependence', False):
                    transformed_ray_directions_expanded[..., -1] = 1

            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            # BATCHED SAMPLE
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            if self.deformnet is not None:
                dx = torch.zeros_like(transformed_points, device=self.device)
                dsigma = torch.zeros((*transformed_points.shape[:2], 1), device=self.device)

                for b in range(batch_size):
                    head = 0
                    while head < transformed_points.shape[1]:
                        tail = head + max_batch_size

                        dx[b:b+1, head:tail], dsigma[b:b+1, head:tail] = self.deformnet.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], trunc_freq_deform[b:b+1], trunc_shifts_deform[b:b+1])[:2]
                        
                        if not no_deformation:
                            transformed_points[b:b+1, head:tail] += dx[b:b+1, head:tail]
                        
                        coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                        if not no_correct:                            
                            coarse_output[b:b+1, head:tail, 3:4] += dsigma[b:b+1, head:tail]

                        head += max_batch_size

            else:
                for b in range(batch_size):
                    head = 0
                    while head < transformed_points.shape[1]:
                        tail = head + max_batch_size
                        coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                        head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            # END BATCHED SAMPLE

            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights, _ = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], \
                                        rgb_clamp_mode=kwargs['rgb_clamp_mode'], noise_std=kwargs['nerf_noise'], white_back = kwargs.get('white_back',False), \
                                        delta_final = kwargs['delta_final'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                     num_steps, det=(randomize==False)).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
                
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                if self.deformnet is not None: 
                    dx_fine = torch.zeros_like(fine_points, device=self.device)
                    dsigma_fine = torch.zeros((*fine_points.shape[:2], 1), device=self.device)
                    # grad_dx_fine = torch.zeros((*fine_points.shape[:2], 3, 3), device=self.device)

                    for b in range(batch_size):
                        head = 0
                        while head < fine_points.shape[1]:
                            tail = head + max_batch_size

                            dx_fine[b:b+1, head:tail], dsigma_fine[b:b+1, head:tail] = self.deformnet.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], trunc_freq_deform[b:b+1], trunc_shifts_deform[b:b+1])[:2]
                            
                            if not no_deformation:
                                fine_points[b:b+1, head:tail] += dx_fine[b:b+1, head:tail]
                            
                            fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail],  truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                            if not no_correct:
                                fine_output[b:b+1, head:tail, 3:4]  += dsigma_fine[b:b+1, head:tail]

                            head += max_batch_size
                else:
                    fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                    for b in range(batch_size):
                        head = 0
                        while head < fine_points.shape[1]:
                            tail = head + max_batch_size
                            fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                            head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                
                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals  = torch.cat([fine_z_vals, z_vals], dim = -2)
            
                _, indices  = torch.sort(all_z_vals, dim=-2)
                all_z_vals  = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))

            else:
                all_outputs = coarse_output
                all_z_vals = z_vals

            pixels, depth, weights, T = fancy_integration(all_outputs, all_z_vals, device=self.device, sigma_only=sigma_only,rgb_only=rgb_only, \
                            white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], rgb_clamp_mode=kwargs['rgb_clamp_mode'], \
                            last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'], \
                            delta_final=kwargs['delta_final'], force_lastPoint_white = kwargs.get('force_lastPoint_white', False))
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map, torch.cat([pitch, yaw], -1), deform_out

    def forward_embedding(self, deform_freq_shift, color_freq_shift, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, stage, alpha,
                            lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0,
                            far_clip=2, sample_dist=None, hierarchical_sample=False,randomize=True,sigma_only=False,rgb_only=-1, 
                            no_correct = False, batch_size=1, 
                            pitch=None, yaw=None, real_pose_meta=None, 
                            return_points=False, points_meta=None,
                            random_rays_ratio=0.25, **kwargs):
        '''
        random_rays_ratio: only part of rays will be calculated gradient
        '''


        H, W = img_size, img_size        

        #------------------- TODO: Input a real pose and bounding box, output a target image
        length_deform = deform_freq_shift.shape[-1]
        length_color = color_freq_shift.shape[-1]
        
        trunc_freq_deform   = deform_freq_shift[...,:length_deform//2]
        trunc_shifts_deform = deform_freq_shift[...,length_deform//2:]

        truncated_frequencies  = color_freq_shift[...,:length_color//2]
        truncated_phase_shifts = color_freq_shift[...,length_color//2:]

        if real_pose_meta is not None:
            pitch = torch.Tensor([0])
            yaw = torch.Tensor([0])

            H, W = real_pose_meta['img_size']
            fov = real_pose_meta['fov']
            cam2world_matrix = torch.Tensor(real_pose_meta['transform_matrix']).clone()
            base_radius = kwargs['original_radius']
            #------ calculate new ray start and ray end
            current_radius = torch.norm(cam2world_matrix[:3,3])
            #------ [original start = (base_radius - 1) / base_radius] , [original end = (base_radius + 2) / base_radius]
            
            cam2world_matrix[:3,3] = cam2world_matrix[:3,3] / base_radius
            ray_residual = (current_radius / base_radius) - 1.0
            ray_start = ray_start + ray_residual
            ray_end = ray_end + ray_residual

            points, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(W, H), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            
            device = self.device
            cam2world_matrix = cam2world_matrix.unsqueeze(0).to(device)

            ray_directions = rays_d_cam
            # ------------- transform points and view-direction given camera pose
            n, num_rays, num_steps, channels = points.shape
            points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
            points_homogeneous[:, :, :, :3] = points
            # should be n x 4 x 4 , n x r^2 x num_steps x 4
            transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
            transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

            # print(cam2world_matrix.shape, torch.zeros((n, num_rays, 4), device=device).shape)
            homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
            homogeneous_origins[:, 3, :] = 1
            # print(homogeneous_origins)
            transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]
            # ------------- [END] transform points and view-direction given camera pose


            transformed_points = transformed_points[..., :3]

            
            if ('bbox' in real_pose_meta) and (real_pose_meta['bbox'] is not None):
                bbox = real_pose_meta['bbox']

                transformed_points = transformed_points.reshape(batch_size, H, W, num_steps, 3)
                transformed_ray_directions = transformed_ray_directions.reshape(batch_size, H, W, 3)
                transformed_ray_origins = transformed_ray_origins.reshape(batch_size, H, W, 3)
                
                z_vals = z_vals.reshape(batch_size, H, W, num_steps, 1)
                # print('======= original z_vals : ', z_vals.shape)

                transformed_points = transformed_points[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:,:]
                transformed_ray_directions = transformed_ray_directions[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:]
                transformed_ray_origins = transformed_ray_origins[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:,]
                
                z_vals = z_vals[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:,:]
                H, W = transformed_points.shape[1:3]


            transformed_ray_directions = transformed_ray_directions.reshape(batch_size, H*W, 3)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)

            transformed_points = transformed_points.reshape(batch_size, H*W*num_steps, 3)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, H*W*num_steps, 3)
            transformed_ray_origins = transformed_ray_origins.reshape(batch_size, H*W, 3)
            z_vals = z_vals.reshape(batch_size, H*W, num_steps, 1)
            # exit(0)

        else:
            if pitch is not None:
                batch_size = pitch.shape[0]

            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(W, H), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean,\
                                                                device=self.device, mode=sample_dist, randomize=randomize, pitch=pitch, yaw=yaw)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, H*W*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, H*W*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1


        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, H*W, num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, H*W, num_steps, 3)

        num_grad_ray   = round(random_rays_ratio * H * W)
        random_perm    = torch.randperm(H*W)
        grad_ray_idx   = random_perm[:num_grad_ray]
        detach_ray_idx = random_perm[num_grad_ray:]
        
        points_grad   = transformed_points[:,grad_ray_idx]
        viewdir_grad  = transformed_ray_directions_expanded[:,grad_ray_idx]

        points_detach  = transformed_points[:,detach_ray_idx]
        viewdir_detach = transformed_ray_directions_expanded[:,detach_ray_idx]
        if self.deformnet is not None:
            dx, dsigma = self.deformnet.forward_with_frequencies_phase_shifts(points_grad, trunc_freq_deform, trunc_shifts_deform)[:2]
            points_grad = points_grad + dx
        coarse_output_grad = self.siren.forward_with_frequencies_phase_shifts(points_grad, truncated_frequencies, truncated_phase_shifts, ray_directions=viewdir_grad, stage=stage, alpha=alpha)
        coarse_output_grad = coarse_output_grad.reshape(batch_size, -1, num_steps, 4)
        if self.deformnet is not None:
            rgb, sigma_t = coarse_output_grad[...,:3], coarse_output_grad[...,3:4]
            sigma = sigma_t + dsigma.view(batch_size, -1, num_steps, 1) 
            coarse_output_grad = torch.cat([rgb, sigma], dim = -1)

        with torch.no_grad():
            if self.deformnet is not None:
                dx, dsigma = self.deformnet.forward_with_frequencies_phase_shifts(points_detach, trunc_freq_deform, trunc_shifts_deform)[:2]
                points_detach = points_detach + dx
            coarse_output_detach = self.siren.forward_with_frequencies_phase_shifts(points_detach, truncated_frequencies, truncated_phase_shifts, ray_directions=viewdir_detach, stage=stage, alpha=alpha)
            coarse_output_detach = coarse_output_detach.reshape(batch_size, -1, num_steps, 4)
            if self.deformnet is not None:
                rgb, sigma_t = coarse_output_detach[...,:3], coarse_output_detach[...,3:4]
                sigma = sigma_t + dsigma.view(batch_size, -1, num_steps, 1) 
                coarse_output_detach = torch.cat([rgb, sigma], dim = -1)
        coarse_output = torch.zeros([batch_size, H*W, num_steps, 4], device=self.device, dtype=coarse_output_grad.dtype)
        coarse_output[:,grad_ray_idx]   = coarse_output_grad
        coarse_output[:,detach_ray_idx] = coarse_output_detach


        if hierarchical_sample:
            with torch.no_grad():
                _, _, weights, _ = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], \
                            rgb_clamp_mode=kwargs['rgb_clamp_mode'], noise_std=kwargs['nerf_noise'], white_back = kwargs.get('white_back',False),\
                            delta_final=kwargs['delta_final'], force_lastPoint_white = kwargs.get('force_lastPoint_white', False))
                            
                weights = weights.reshape(batch_size * H*W, num_steps) + 1e-5
                z_vals = z_vals.reshape(batch_size * H*W, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, H*W, num_steps, 1)


                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                    num_steps, det=kwargs.get('sample_no_random', False)).detach()
                
                fine_z_vals = fine_z_vals.reshape(batch_size, H*W, num_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, H*W, num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions = torch.zeros_like(transformed_ray_directions)
                    transformed_ray_directions[..., -1] = -1
                    viewdir_grad  = transformed_ray_directions_expanded[:,grad_ray_idx]
                    viewdir_detach = transformed_ray_directions_expanded[:,detach_ray_idx]

            #---------------------------------------------------------------------------------------------
            fine_points_grad   = fine_points[:,grad_ray_idx]
            fine_points_detach = fine_points[:,detach_ray_idx]
            
            if self.deformnet is not None:
                dx_fine, dsigma_fine = self.deformnet.forward_with_frequencies_phase_shifts(fine_points_grad, trunc_freq_deform, trunc_shifts_deform)[:2]
                fine_points_grad = fine_points_grad + dx_fine
            fine_output_grad = self.siren.forward_with_frequencies_phase_shifts(fine_points_grad, truncated_frequencies, truncated_phase_shifts, ray_directions=viewdir_grad, stage=stage, alpha=alpha).reshape(batch_size, -1, num_steps, 4)
            if self.deformnet is not None:
                rgb_fine, sigma_fine_t = fine_output_grad[...,:3], fine_output_grad[...,3:4]
                sigma_fine = sigma_fine_t + dsigma_fine.view(batch_size, -1, num_steps, 1)
                fine_output_grad = torch.cat([rgb_fine, sigma_fine], dim = -1)

            with torch.no_grad():
                if self.deformnet is not None:
                    dx_fine, dsigma_fine = self.deformnet.forward_with_frequencies_phase_shifts(fine_points_detach, trunc_freq_deform, trunc_shifts_deform)[:2]
                    fine_points_detach = fine_points_detach + dx_fine
                fine_output_detach = self.siren.forward_with_frequencies_phase_shifts(fine_points_detach, truncated_frequencies, truncated_phase_shifts, ray_directions=viewdir_detach, stage=stage, alpha=alpha).reshape(batch_size, -1, num_steps, 4)
                if self.deformnet is not None:
                    rgb_fine, sigma_fine_t = fine_output_detach[...,:3], fine_output_detach[...,3:4]
                    sigma_fine = sigma_fine_t + dsigma_fine.view(batch_size, -1, num_steps, 1)
                    fine_output_detach = torch.cat([rgb_fine, sigma_fine], dim = -1)              

            fine_output = torch.zeros([batch_size, H*W, num_steps, 4], device=self.device, dtype=fine_output_grad.dtype)
            fine_output[:,grad_ray_idx]   = fine_output_grad
            fine_output[:,detach_ray_idx] = fine_output_detach

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals  = torch.cat([fine_z_vals, z_vals], dim = -2)

            _, indices  = torch.sort(all_z_vals, dim=-2)
            all_z_vals  = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))

        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        pixels, depth, weights, T = fancy_integration(all_outputs, all_z_vals, device=self.device, sigma_only=sigma_only,rgb_only=rgb_only,\
                        white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], rgb_clamp_mode=kwargs['rgb_clamp_mode'], \
                        last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'], \
                        delta_final=kwargs['delta_final'], force_lastPoint_white=kwargs.get('force_lastPoint_white', False))

        pixels = pixels.reshape((batch_size, H, W, 3)).permute(0, 3, 1, 2) * 2 - 1
        depth = depth.reshape(batch_size, H, W)


        extra_output = {}
        # extra_output['all_output'] = all_outputs
        # extra_output['T'] = T
        if return_points:
            ret_z_vals  = all_z_vals

            ret_points  = torch.cat([fine_points.view(batch_size, -1, num_steps, 3), \
                                    transformed_points.view(batch_size, -1, num_steps, 3)], dim=-2)
            ret_viewdir = transformed_ray_directions_expanded.view(batch_size, -1, num_steps, 3)
            ret_viewdir = torch.cat([ret_viewdir, ret_viewdir], dim=-2)
            
            ret_points  = torch.gather(ret_points, -2, indices.expand(-1,-1,-1,3))
            ret_viewdir = torch.gather(ret_viewdir, -2, indices.expand(-1,-1,-1,3))

            points_meta = {}
            points_meta['ret_z_vals']  = ret_z_vals
            points_meta['ret_points']  = ret_points
            points_meta['ret_viewdir'] = ret_viewdir
            points_meta['ret_depth']   = depth.detach().reshape(batch_size, -1, 1)
            extra_output['points_meta'] = points_meta

        return pixels, depth, torch.cat([pitch, yaw], -1), extra_output

    def forward_with_frequencies(self, deform_freq_shift, color_freq_shift, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, stage, alpha,
                            psi=1.0, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0,
                            far_clip=2, sample_dist=None, hierarchical_sample=False,randomize=True,sigma_only=False,rgb_only=-1, 
                            debug=False, no_correct = False, batch_size=1, 
                            pitch=None, yaw=None, real_pose_meta=None, 
                            return_points=False, points_meta=None, **kwargs):

        
        H, W = img_size, img_size        

        #------------------- TODO: Input a real pose and bounding box, output a target image

        length_deform = deform_freq_shift.shape[-1]
        length_color = color_freq_shift.shape[-1]
        
        trunc_freq_deform   = deform_freq_shift[...,:length_deform//2]
        trunc_shifts_deform = deform_freq_shift[...,length_deform//2:]

        truncated_frequencies  = color_freq_shift[...,:length_color//2]
        truncated_phase_shifts = color_freq_shift[...,length_color//2:]

        if points_meta is not None:
            all_points  = points_meta['ret_points']
            all_viewdir = points_meta['ret_viewdir']
            all_z_vals  = points_meta['ret_z_vals']

            pixels, pose_info, T, deform_out = self.forward_with_freq_points(trunc_freq_deform, trunc_shifts_deform, truncated_frequencies, truncated_phase_shifts, \
                                    all_points, all_viewdir, all_z_vals, pitch=pitch, yaw=yaw, **kwargs)
            pixels = pixels.reshape((batch_size, 3, img_size, img_size))
            return pixels, pose_info, T, deform_out

        if real_pose_meta is not None:
            pitch = torch.Tensor([0])
            yaw = torch.Tensor([0])

            H, W = real_pose_meta['img_size']
            fov = real_pose_meta['fov']
            cam2world_matrix = torch.Tensor(real_pose_meta['transform_matrix'])
            base_radius = kwargs['original_radius']
            #------ calculate new ray start and ray end
            current_radius = torch.norm(cam2world_matrix[:3,3])
            #------ [original start = (base_radius - 1) / base_radius] , [original end = (base_radius + 2) / base_radius]
            
            cam2world_matrix[:3,3] = cam2world_matrix[:3,3] / base_radius
            ray_residual = (current_radius / base_radius) - 1.0
            ray_start = ray_start + ray_residual
            ray_end = ray_end + ray_residual

            points, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(W, H), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            
            device = self.device
            cam2world_matrix = cam2world_matrix.unsqueeze(0).to(device)

            # print('======= cam2world_matrix : ', cam2world_matrix.shape)


            ray_directions = rays_d_cam
            # ------------- transform points and view-direction given camera pose
            n, num_rays, num_steps, channels = points.shape
            points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
            points_homogeneous[:, :, :, :3] = points
            # should be n x 4 x 4 , n x r^2 x num_steps x 4
            transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
            transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

            # print(cam2world_matrix.shape, torch.zeros((n, num_rays, 4), device=device).shape)
            homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
            homogeneous_origins[:, 3, :] = 1
            # print(homogeneous_origins)
            transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]
            # ------------- [END] transform points and view-direction given camera pose


            transformed_points = transformed_points[..., :3]

            
            if ('bbox' in real_pose_meta) and (real_pose_meta['bbox'] is not None):
                bbox = real_pose_meta['bbox']

                # print('======= transformed_points : ', transformed_points.shape)

                transformed_points = transformed_points.reshape(batch_size, H, W, num_steps, 3)
                transformed_ray_directions = transformed_ray_directions.reshape(batch_size, H, W, 3)
                transformed_ray_origins = transformed_ray_origins.reshape(batch_size, H, W, 3)
                
                z_vals = z_vals.reshape(batch_size, H, W, num_steps, 1)
                # print('======= original z_vals : ', z_vals.shape)

                # exit(0)
                transformed_points = transformed_points[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:,:]
                transformed_ray_directions = transformed_ray_directions[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:]
                transformed_ray_origins = transformed_ray_origins[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:,]
                
                z_vals = z_vals[:,bbox[1]:bbox[3], bbox[0]:bbox[2],:,:]
            
                H, W = transformed_points.shape[1:3]

            transformed_ray_directions = transformed_ray_directions.reshape(batch_size, H*W, 3)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)

            transformed_points = transformed_points.reshape(batch_size, H*W*num_steps, 3)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, H*W*num_steps, 3)
            transformed_ray_origins = transformed_ray_origins.reshape(batch_size, H*W, 3)
            z_vals = z_vals.reshape(batch_size, H*W, num_steps, 1)
            # exit(0)

        else:
            if pitch is not None:
                batch_size = pitch.shape[0]

            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(W, H), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean,\
                                                                device=self.device, mode=sample_dist, randomize=randomize, pitch=pitch, yaw=yaw)
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, H*W*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, H*W*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1

        # BATCHED SAMPLE
        coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
        if self.deformnet is not None:
            dx = torch.zeros_like(transformed_points, device=self.device)
            dsigma = torch.zeros((*transformed_points.shape[:2], 1), device=self.device)

            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size

                    dx[b:b+1, head:tail], dsigma[b:b+1, head:tail] = self.deformnet.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], trunc_freq_deform[b:b+1], trunc_shifts_deform[b:b+1])                    
                    input_points = transformed_points[b:b+1, head:tail] + dx[b:b+1, head:tail]
                    tmp_output = self.siren.forward_with_frequencies_phase_shifts(input_points, truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                    if not no_correct:                      
                        # coarse_output[b:b+1, head:tail, 3:4]  +=  dsigma[b:b+1, head:tail]
                        coarse_output[b:b+1, head:tail,  :3] = tmp_output[..., :3]
                        coarse_output[b:b+1, head:tail, 3:4] = tmp_output[...,3:4] + dsigma[b:b+1, head:tail]
                    else:
                        coarse_output[b:b+1, head:tail] = tmp_output

                    head += max_batch_size
        else:
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                    head += max_batch_size

        coarse_output = coarse_output.reshape(batch_size, H*W, num_steps, 4)
        # END BATCHED SAMPLE



        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, H * W, num_steps, 3)
                _, _, weights, _ = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], rgb_clamp_mode=kwargs['rgb_clamp_mode'], noise_std=kwargs['nerf_noise'],delta_final=kwargs['delta_final'], white_back = kwargs.get('white_back',False))

                weights = weights.reshape(batch_size * H * W, num_steps) + 1e-5
                z_vals = z_vals.reshape(batch_size * H * W, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, H * W, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                    num_steps, det=(randomize==False)).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, H * W, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, H*W*num_steps, 3)
                #### end new importance sampling

                if debug:
                    deform_out['raw_fine_points'] = fine_points.clone()

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
            # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
            
            # BATCHED SAMPLE
            fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
            if self.deformnet is not None: 
                dx_fine = torch.zeros_like(fine_points, device=self.device)
                dsigma_fine = torch.zeros((*fine_points.shape[:2], 1), device=self.device)
                # grad_dx_fine = torch.zeros((*fine_points.shape[:2], 3, 3), device=self.device)

                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        dx_fine[b:b+1, head:tail], dsigma_fine[b:b+1, head:tail] = self.deformnet.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], trunc_freq_deform[b:b+1], trunc_shifts_deform[b:b+1])
                        input_points = fine_points[b:b+1, head:tail] + dx_fine[b:b+1, head:tail]
                        tmp_output = self.siren.forward_with_frequencies_phase_shifts(input_points,  truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                        if not no_correct:
                            fine_output[b:b+1, head:tail,  :3] = tmp_output[...,:3] 
                            fine_output[b:b+1, head:tail, 3:4] = tmp_output[...,3:4] +  dsigma_fine[b:b+1, head:tail]
                        else:
                            fine_output[b:b+1, head:tail] = tmp_output

                        head += max_batch_size
            else:
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                        head += max_batch_size

            fine_output = fine_output.reshape(batch_size, H*W, num_steps, 4)
            # END BATCHED SAMPLE

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        pixels, depth, weights, T = fancy_integration(all_outputs, all_z_vals, device=self.device, sigma_only=sigma_only,rgb_only=rgb_only, \
                                            white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], rgb_clamp_mode=kwargs['rgb_clamp_mode'], \
                                            last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'], \
                                            delta_final=kwargs['delta_final'], force_lastPoint_white = kwargs.get('force_lastPoint_white', False))
        depth_map = depth.reshape(batch_size, H, W).contiguous()

        pixels = pixels.reshape((batch_size, H, W, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        extra_output = {}
        extra_output['all_output'] = all_outputs
        extra_output['T'] = T


        if return_points:
            ret_z_vals  = all_z_vals

            ret_points  = torch.cat([fine_points.view(batch_size, -1, num_steps, 3), \
                                    transformed_points.view(batch_size, -1, num_steps, 3)], dim=-2)
            ret_viewdir = transformed_ray_directions_expanded.view(batch_size, -1, num_steps, 3)
            ret_viewdir = torch.cat([ret_viewdir, ret_viewdir], dim=-2)
            
            ret_points  = torch.gather(ret_points, -2, indices.expand(-1,-1,-1,3))
            ret_viewdir = torch.gather(ret_viewdir, -2, indices.expand(-1,-1,-1,3))

            points_meta = {}
            points_meta['ret_z_vals']  = ret_z_vals
            points_meta['ret_points']  = ret_points
            points_meta['ret_viewdir'] = ret_viewdir
            points_meta['ret_depth']   = depth.reshape(batch_size, -1, 1)
            extra_output['points_meta'] = points_meta

        return pixels, depth_map, torch.cat([pitch, yaw], -1), extra_output