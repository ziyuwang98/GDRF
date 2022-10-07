import numpy as np
import torch

def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z

def normalize_img(tensor_img, img_range = (-1.,1.)):
    img = tensor_img.detach().permute(0, 2, 3, 1).squeeze() #(B, H, W, 3)
    img = (img - img_range[0])/(img_range[1]-img_range[0])
    img = (img.cpu().numpy()*255).astype(np.uint8)
    return img

def normalize_depth(tensor_depth, max_depth=None):
    depth = tensor_depth.detach().squeeze() #(B, H, W)
    B, H, W = depth.shape

    depth =  depth.view(B, H*W)

    if max_depth is not None:   #set Nan to the max depth
        depth[depth!=depth] = max_depth

    depth_min, _ = torch.min(depth, dim=1, keepdims=True)
    depth_max, _ = torch.max(depth, dim=1, keepdims=True)

    depth = (depth - depth_min) / (depth_max-depth_min)

    depth = (depth.view(B, H, W).cpu().numpy()*255).astype(np.uint8)
    return depth

def process_training_status_img(full_imgs, full_depth, noCor_imgs, noCor_depth, temp_imgs, temp_depth, img_size, max_depth):
    all_img   = torch.cat([full_imgs.view(4,4,3,img_size,img_size),
                        noCor_imgs.view(4,4,3,img_size,img_size),
                            temp_imgs.view(4,4,3,img_size,img_size)], dim=1)
    all_depth = torch.cat([full_depth.view(4,4,img_size,img_size),
                        noCor_depth.view(4,4,img_size,img_size),
                            temp_depth.view(4,4,img_size,img_size)], dim=1)
    all_img   = normalize_img(all_img.reshape(-1, 3, img_size, img_size)).reshape(4, 12, img_size, img_size, 3)

    all_depth = normalize_depth(all_depth.reshape(-1,img_size,img_size), max_depth=max_depth).reshape(4, 12, img_size, img_size, 1)
    all_depth = all_depth.repeat(3, -1)
    final_img = np.zeros([8*img_size, 12*img_size, 3], dtype=np.uint8)
    for tmp_i in range(4):
        for tmp_j in range(12):
            final_img[tmp_i*img_size:(tmp_i+1)*img_size,  tmp_j*img_size:(tmp_j+1)*img_size] = all_img[tmp_i, tmp_j]
            final_img[(tmp_i+4)*img_size:(tmp_i+5)*img_size,  tmp_j*img_size:(tmp_j+1)*img_size] = all_depth[tmp_i, tmp_j]
    return final_img