import math
import numpy as np
import torch
import torch.nn.functional as F
import random
from .math_utils_torch import *


def fancy_integration(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, 
                      clamp_mode=None, rgb_clamp_mode=None, fill_mode=None, eps=1e-3, sigma_only=False, 
                      rgb_only=-1, delta_final=1e10, mean_delta_final=False, 
                      ray_end_delta_final=False, ray_end=None, raw_depth=False, return_alpha=False, 
                      force_lastPoint_white=False):

    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]

    if force_lastPoint_white:
        rgbs[...,-1,:] = 1.0

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    
    if ray_end_delta_final:
        delta_inf = torch.abs(z_vals[:, :, -1:] - ray_end)
    elif mean_delta_final:
        delta_inf = torch.mean(deltas, dim=-2, keepdim=True)
    else:
        delta_inf = delta_final * torch.ones_like(deltas[:, :, :1])
    
    deltas = torch.cat([deltas, delta_inf], -2)
    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if not rgb_only==-1:
        sigmas[z_vals<rgb_only] = -1e5
        sigmas[z_vals>=rgb_only] = 1e5

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"
    
    if rgb_clamp_mode == 'sigmoid':
        pass
    elif rgb_clamp_mode == 'widen_sigmoid':
        rgbs = rgbs*(1+2*eps) - eps
    else:
        raise "Need to choose rgb clamp mode"
    
    if sigma_only:
        rgbs = torch.zeros_like(rgbs)
        rgbs[...,2] += 0.5
    
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    T = torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights = alphas * T

    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)

    if raw_depth:
        depth_final = torch.sum(weights * z_vals, -2)
    else:
        depth_final = torch.sum(weights * z_vals, -2)/weights_sum

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    if return_alpha:
        return rgb_final, depth_final, weights, T, alphas
    else:
        return rgb_final, depth_final, weights, T

def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, ray directions in camera space."""
    W, H = resolution
    if W == H:
        # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
        # Y is flipped to follow image memory layouts.
        x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                              torch.linspace(1, -1, H, device=device))
        x = x.T.flatten()
        y = y.T.flatten()
        z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)
        rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))
    else:
        focal = W / (2*np.tan(fov*np.pi/360.0))
        K = torch.Tensor([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ]).to(device)
        
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t().to(device)
        j = j.t().to(device)
        rays_d_cam = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i, device=device)], -1)
        rays_d_cam = rays_d_cam / torch.norm(rays_d_cam, dim=-1, keepdim=True)
        rays_d_cam = rays_d_cam.reshape(-1, 3)


    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return points, z_vals, rays_d_cam

def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals

def perturb_points_2(points, z_vals, ray_directions, device, spread):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = torch.randn(z_vals.shape, device=device) * spread
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2).contiguous()
    return points, z_vals

def transform_sampled_points(points, z_vals, ray_directions, device, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5,\
                             mode='normal', randomize=True, pitch=None, yaw=None):
    n, num_rays, num_steps, channels = points.shape

    if randomize:
        points, z_vals = perturb_points(points, z_vals, ray_directions, device)

    if (pitch is not None) and (yaw is not None):
        r = 1.0
        camera_origin = torch.zeros((n, 3), device=device)
        camera_origin[:, 0:1] = r*torch.sin(pitch) * torch.cos(yaw)
        camera_origin[:, 2:3] = r*torch.sin(pitch) * torch.sin(yaw)
        camera_origin[:, 1:2] = r*torch.cos(pitch)
    else:
        camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)

    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)
    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw


def transform_sampled_points_with_pose(points, z_vals, ray_directions, device, phi, theta, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal',randomize=True):
    n, num_rays, num_steps, channels = points.shape

    if randomize:
        points, z_vals = perturb_points(points, z_vals, ray_directions, device)

    camera_origin, pitch, yaw = get_camera_positions_with_pose(phi=phi, theta=theta, n=points.shape[0], r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points
    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)
    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw


def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """Samples n random locations along a sphere of radius r. Uses a gaussian distribution for pitch and yaw"""
    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
            
    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi # convert from radians to [0,1]
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)
    elif mode == 'sphere_random':
        pos = torch.randn((n,3), device=device, dtype=torch.float)
        pos = pos / torch.norm(pos, dim=-1, keepdim=True)
        
        theta = torch.atan2(-pos[:,0], pos[:,2]).unsqueeze(-1)
        phi = torch.atan(-pos[:,1] / torch.norm(pos[:,::2], dim=-1)).unsqueeze(-1)
        theta = theta + math.pi * 0.5
        phi = phi + math.pi * 0.5

    else:
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta


def get_camera_positions_with_pose(device, phi, theta,  n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    
    theta = (torch.sigmoid(theta)-0.5) * 2 * horizontal_stddev + horizontal_mean
    phi = (torch.sigmoid(phi)-0.5) * 2 * vertical_stddev + vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta

def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:   #TODO: set True
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)

    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples