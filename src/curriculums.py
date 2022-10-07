import math

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    last_epoch = 0
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step:
            last_epoch = curriculum_step
    return last_epoch

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    # step = get_current_step(curriculum, epoch)
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict

CHAIR_128 = {
    0:       {'batch_size': 4, 'num_steps': 36, 'img_size': 64,  'batch_split': 4, 'num_regions': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CHAIR64_1_VIEW'},
    95000:   {'batch_size': 4, 'num_steps': 64, 'img_size': 64,  'batch_split': 4, 'num_regions': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CHAIR64_1_VIEW'},
    95100:   {'batch_size': 4, 'num_steps': 64, 'img_size': 128, 'batch_split': 4, 'num_regions': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CHAIR128_1_VIEW'},
    int(110000): {},

    'generator_module': 'generators',

    #--------------------------- Deformation Setting
    'use_deformnet': True,
    'deformnet': 'ImplicitDeformation3D',
    'deformnet_module': 'deformnets', 
    'deformnet_model' : 'SPATIAL_SIREN_DEFORM',

    ## regs
    'correct_reg': 1e-2,
    'correct_scale': 100.0, # the scale for the raw output of the density correction

    'reg_ray_sample_ratio': 0.05,  # [deform and normal] random sample rays for calculating regularization
    'reg_near_depth_ratio': 0.2, # [normal] the points near the depth will be used for calculate normal regularization
    
    'deform_smooth_reg': 1.0,
    'deform_rigid_reg': 0.1,
    'normal_consist_reg': 5.0,
    #--------------------------- Deformation Setting (End)

    'use_mask': True,
    'white_back': True,
    'real_pose': True,

    'lock_view_dependence': False,
    'img_flip': False,

    'fov': 50,
    'ray_start': 0.6, 
    'ray_end':   1.4,

    'fade_steps': 10000,
    'h_stddev': math.pi,
    'h_mean': 0.,
    'v_stddev': 0.2618,
    'v_mean': 1.070796,
    'sample_dist': 'uniform',
    
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'hidden_dim': 256,
    'hidden_dim_sample': 128,
    'grad_clip': 0.3,
    'model': 'SPATIAL_SIREN_TEMPLATE_DEFERDIR',
    'generator': 'ImplicitGenerator3d',

    'discriminator': 'ProgressiveEncoderDiscriminatorAntiAlias',
    'clamp_mode': 'softplus',
    'rgb_clamp_mode': 'widen_sigmoid',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
    'use_pix_noise': False,
    'phase_noise': False,
    'delta_final': 1e10,
    'equal_lr': 1,
}

CAR_128 = {
    0:     {'batch_size': 4, 'num_steps': 36, 'img_size': 64, 'batch_split':  1, 'num_regions': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CAR64_3_VIEW'},
    90000: {'batch_size': 4, 'num_steps': 64, 'img_size': 64, 'batch_split':  4, 'num_regions': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CAR64_3_VIEW'},
    90100: {'batch_size': 4, 'num_steps': 64, 'img_size': 128, 'batch_split': 4, 'num_regions': 4, 'gen_lr': 1e-5, 'disc_lr': 1e-4, 'dataset': 'CAR128_3_VIEW'},
    int(130000): {},

    'generator_module': 'generators',

    #--------------------------- Deformation Setting
    'use_deformnet': True,
    'deformnet': 'ImplicitDeformation3D',
    'deformnet_module': 'deformnets', 
    'deformnet_model' : 'SPATIAL_SIREN_DEFORM',

    ## regs
    'correct_reg': 1e-1,
    'correct_scale': 100.0, # the scale for the raw output of the density correction
    'correct_random': True, # only apply correction reg. on random sampled rays by reg_ray_sample_ratio

    'reg_ray_sample_ratio': 0.05,  # [deform and normal] random sample rays for calculating regularization
    'reg_near_depth_ratio': 0.2, # [normal] the points near the depth will be used for calculate normal regularization
    
    'deform_smooth_reg': 1.0,
    'deform_rigid_reg': 0.05,
    'normal_consist_reg': 5.0,
    #--------------------------- Deformation Setting (End)

    'use_mask': True,
    'white_back': True,
    'real_pose': True,

    'lock_view_dependence': False,
    'img_flip': False,

    'fov': 30,
    'ray_start': 0.75, 
    'ray_end':   1.25,

    'fade_steps': 10000,
    'h_stddev': math.pi,
    'h_mean': 0.,
    'v_stddev': 0.2618,
    'v_mean': 1.2217,
    'sample_dist': 'uniform',
    
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'hidden_dim': 256,
    'hidden_dim_sample': 128,
    'grad_clip': 0.3,
    'model': 'SPATIAL_SIREN_TEMPLATE_DEFERDIR',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminatorAntiAlias',
    'clamp_mode': 'softplus',
    'rgb_clamp_mode': 'widen_sigmoid',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
    'use_pix_noise': False,
    'phase_noise': False,
    'delta_final': 1e10,
    'equal_lr': 1,
}

CARLA_128 = {
    0:     {'batch_size': 4, 'num_steps': 36, 'img_size': 64,  'batch_split': 4, 'num_regions': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CARLA64'},
    45000: {'batch_size': 4, 'num_steps': 48, 'img_size': 64,  'batch_split': 4, 'num_regions': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CARLA64'},
    45100: {'batch_size': 4, 'num_steps': 48, 'img_size': 128, 'batch_split': 4, 'num_regions': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'CARLA128'},
    int(110000): {},

    'generator_module': 'generators',

    #--------------------------- Deformation Setting
    'use_deformnet': True,
    'deformnet': 'ImplicitDeformation3D',
    'deformnet_module': 'deformnets', 
    'deformnet_model' : 'SPATIAL_SIREN_DEFORM',

    ## regs
    'correct_reg': 1e-1,
    'correct_scale': 100.0, # the scale for the raw output of the density correction

    'reg_ray_sample_ratio': 0.05,  # [deform and normal] random sample rays for calculating regularization
    'reg_near_depth_ratio': 0.2, # [normal] the points near the depth will be used for calculate normal regularization
    
    'deform_smooth_reg': 1.0,
    'deform_rigid_reg': 0.1,
    'normal_consist_reg': 5.0,
    #--------------------------- Deformation Setting (End)

    'use_mask': True,
    'white_back': True,
    'real_pose': True,

    'lock_view_dependence': True,
    'img_flip': False,

    'fov': 30,
    'ray_start': 0.7,
    'ray_end':   1.3,

    'fade_steps': 10000,
    'h_stddev': math.pi,
    'h_mean': math.pi*0.5,
    'v_stddev': math.pi/4 * 85/90,
    'v_mean': math.pi/4 * 85/90,
    'sample_dist': 'spherical_uniform',
    
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'hidden_dim': 256,
    'hidden_dim_sample': 128,
    'grad_clip': 0.3,
    'model': 'SPATIAL_SIREN_TEMPLATE_DEFERDIR',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminatorAntiAlias',
    'clamp_mode': 'softplus',
    'rgb_clamp_mode': 'widen_sigmoid',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
    'use_pix_noise': False,
    'phase_noise': False,
    'delta_final': 1e10,
    'equal_lr': 1,
}