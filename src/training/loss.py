import torch
import torch.nn.functional as F

#---- correction
def minimal_correction_regularization(dsigma):
    return torch.abs(dsigma).mean()

#---- deformation
def deform_smooth_regularization(grad_dx):
    return grad_dx.norm(dim=-1).mean()

def deform_rigid_regularization(grad_dx):
    # calculate deformation gradient from displacement gradient
    B, N = grad_dx.shape[:2]
    jacob = grad_dx.clone().reshape(B, N, 9)
    jacob[:,:,::grad_dx.shape[2]+1] += 1.0 
    jacob = jacob.view(grad_dx.shape)
    jtj   = jacob.permute(0,1,3,2) @ jacob
    loss  = jtj.view(B,N,9) - torch.eye(3, device=jtj.device).view(1,1,9)
    loss  = 0.5 * torch.norm(loss, p = 'fro', dim=-1)
    return loss.mean()

#---- normal consistency
def normal_consistent_regularization(grad_sigma, grad_sigma_temp):
    loss = 1. - F.cosine_similarity(grad_sigma_temp, grad_sigma, dim=-1)
    return loss.mean()