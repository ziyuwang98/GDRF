import os
import sys
import importlib
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

sys.path.append('..')
from discriminators import discriminators
from siren import siren

def set_deformnet(metadata, device, opt):
    if not 'deformnet_module' in metadata:
        deformnets = importlib.import_module('deformnets.deformnets')
    else:
        deformnets = importlib.import_module('deformnets.' + metadata['deformnet_module'])

    deformnet_core = getattr(siren, metadata['deformnet_model'])
    deformnet = getattr(deformnets, metadata['deformnet'])(deformnet_core, metadata['latent_dim'], **metadata)
    return deformnet

def set_generator(metadata, device, opt, deformnet_ins=None, bgnet_ins=None):
    if not 'generator_module' in metadata:
        generators = importlib.import_module('generators.generators')
    else:
        generators = importlib.import_module('generators.'+metadata['generator_module'])

    generator_core = getattr(siren, metadata['model'])
    generator = getattr(generators, metadata['generator'])(generator_core, metadata['latent_dim'], deformnet_ins = deformnet_ins, bgnet_ins=bgnet_ins, **metadata)

    if opt.load_dir != '':
        generator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_generator.pth'%opt.set_step), map_location='cpu'))

    generator = generator.to(device)
    
    if opt.load_dir != '':
        ema = torch.load(os.path.join(opt.load_dir, 'step%06d_ema.pth'%opt.set_step), map_location=device)
        ema2 = torch.load(os.path.join(opt.load_dir, 'step%06d_ema2.pth'%opt.set_step), map_location=device)
    else:
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)
    
    return generator, ema, ema2

def set_discriminator(metadata,device,opt):
    discriminator = getattr(discriminators, metadata['discriminator'])(metadata['latent_dim'],**metadata)
    if opt.load_dir != '':
        discriminator.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_discriminator.pth'%opt.set_step), map_location='cpu'))
    discriminator = discriminator.to(device)    
    return discriminator

def set_optimizer_G(generator_ddp, metadata, opt):
    if 'equal_lr' not in metadata:
        metadata['equal_lr'] = 5e-2
    if 'sample_lr' not in metadata:
        metadata['sample_lr'] = 1
    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_param_names += [name for name, _ in generator_ddp.module.deformnet.siren.mapping_network.named_parameters()]
        
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n.replace('module.siren.mapping_network.','') in mapping_network_param_names \
                                    or n.replace('module.deformnet.siren.mapping_network.','') in mapping_network_param_names]

        generator_parameters = [p for n, p in generator_ddp.named_parameters() if (n.replace('module.siren.mapping_network.','') not in mapping_network_param_names) \
                                    and (n.replace('module.deformnet.siren.mapping_network.','') not in mapping_network_param_names)]
        optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                        {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*metadata['equal_lr']}],
                                       lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_G.pth'%opt.set_step), map_location='cpu'))
    
    return optimizer_G

def set_optimizer_D(discriminator_ddp, metadata, opt):
    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_D.pth'%opt.set_step), map_location='cpu'))
    
    return optimizer_D

def set_learning_rate_G(optimizer_G, metadata):
    if 'equal_lr' not in metadata:
        metadata['equal_lr'] = 5e-2
    for param_group in optimizer_G.param_groups:
        if param_group.get('name', None) == 'mapping_network':
            param_group['lr'] = metadata['gen_lr'] * metadata['equal_lr']
        elif param_group.get('name', None) == 'sample_network':
            param_group['lr'] = metadata['gen_lr'] * metadata['sample_lr']
        else:
            param_group['lr'] = metadata['gen_lr']
        param_group['betas'] = metadata['betas']
        param_group['weight_decay'] = metadata['weight_decay']

def set_learning_rate_D(optimizer_D, metadata):
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = metadata['disc_lr']
        param_group['betas'] = metadata['betas']
        param_group['weight_decay'] = metadata['weight_decay']