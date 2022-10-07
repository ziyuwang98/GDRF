"""Implicit deformation network"""

import torch.nn as nn
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

class ImplicitDeformation3D(nn.Module):
    def __init__(self, siren, z_dim, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None, **kwargs)

    def forward(self, x, z, **kwargs):
        '''
        Input: 
            x : (B, N, 3)   3D points
            z : (B, K)    k-dim latent vector
        Ouput:
            dx: (B, N, 3)   3D displacements
            dsigma: (B, N, 1) density correction
        '''
        return self.siren.foward(x,z,**kwargs)

    def mapping_network(self, z):
        '''
        Input: 
            x : (B, N, 3)   3D points
            z : (B, K)    k-dim latent vector
        Ouput: 
            frequencies
            phase_shifts
        '''
        frequencies, phase_shifts = self.siren.mapping_network(z)
        return frequencies, phase_shifts

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, **kwargs):
        return self.siren.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, **kwargs)

    def calc_disp_gradient(self, x, dx, **kwargs):
        '''
        Input: 
            x : (B, N, 3)   3D points
            dx: (B, N, 3)   3D displacement
        Ouput: 
            grad_dx : (B, N, 3, 3)   the displacement gradient
        '''
        u = dx[..., 0]
        v = dx[..., 1]
        w = dx[..., 2]
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(w, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_dx = torch.stack([grad_u,grad_v,grad_w],dim=2)

        return grad_dx