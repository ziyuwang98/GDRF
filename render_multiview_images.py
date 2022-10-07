
import sys
import os
import copy
import re
import argparse

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

sys.path.append('./src')
from generators import generators
from siren import siren
import curriculums
from training.model_setup import set_deformnet


def normalize_img(tensor_img, img_range = (-1.,1.)):
    img = tensor_img.detach().permute(0, 2, 3, 1).squeeze() #(B, H, W, 3)
    img = (img - img_range[0])/(img_range[1]-img_range[0])
    img = (img.cpu().numpy()*255).astype(np.uint8)
    return img

def normalize_depth(tensor_depth):
    depth = tensor_depth.detach().squeeze() #(B, H, W)
    B, H, W = depth.shape

    depth =  depth.view(B, H*W)
    depth_min, _ = torch.min(depth, dim=1, keepdims=True)
    depth_max, _ = torch.max(depth, dim=1, keepdims=True)
    depth = (depth - depth_min) / (depth_max-depth_min)

    depth = (depth.view(B, H, W).cpu().numpy()*255).astype(np.uint8)
    return depth


def get_ellipse_pose(h_mean, v_mean, h_stddev, v_stddev, num_view, scale=1.0):
    angles = [2*np.pi*tmp for tmp in np.linspace(0.,1.,num_view)]
    ellipse_a = h_stddev * scale
    ellipse_b = v_stddev * scale
    
    v_h_l = []  # (y, x)
    for angle in angles:
        h = h_mean + ellipse_a * np.cos(angle)
        v = v_mean + ellipse_b * np.sin(angle)
        v_h_l.append((v,h))
    v_h_l = np.array(v_h_l)

    return v_h_l

def parse_idx_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def get_seed_string(code_seed_l):
    code_string = ''
    for i, seed_id in enumerate(code_seed_l):
        code_string += f'{seed_id}'
        if i != (len(code_seed_l)-1):
            code_string += ','
    return code_string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='./pretrained_models/CHAIR_128/generator.pth')
    parser.add_argument('--output_dir', type=str, default='render_multiview')
    parser.add_argument('--gpu', type=int, default='0', help='The gpu id')
    parser.add_argument('--curriculum', type=str, default='CHAIR_128')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--num_view', type=int, default=75, help='The number of render views for each object')
    parser.add_argument('--num_steps', type=int, default=96)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--psi', type=float, default=0.7)
    parser.add_argument('--save_raw', action='store_true', help='Save all the raw output image')
    parser.add_argument('--render_template', action='store_true',help='Render without deform-net')
    parser.add_argument('--no_correction', action='store_true', help='Render without correction field')
    parser.add_argument('--input_seed_list', type=parse_idx_range, help='The seed list for both shape and the appearance')
    parser.add_argument('--input_appear_seed', default = -1, type = int, help='The seed for the appearance code')
    parser.add_argument('--ellipse_pose', action='store_true', help='Pose trajectory is an ellipse.')

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    curriculum_name = opt.curriculum

    no_correct_prefix = ''
    if opt.no_correction:
        no_correct_prefix = 'noCorrection_'

    appear_trans_prefix = ''
    if opt.input_appear_seed > 0:
        appear_trans_prefix = f'colorSeed_{opt.input_appear_seed}_'

    ellipse_prefix = ''
    if opt.ellipse_pose:
        ellipse_prefix = 'ellipse_'

    if opt.render_template:
        no_correct_prefix = ''
        opt.no_correction = False
        output_dir = os.path.join(opt.output_dir, curriculum_name, f'TEMPLATE_{ellipse_prefix}{appear_trans_prefix}_psi_{opt.psi}')
    else:
        output_dir = os.path.join(opt.output_dir, curriculum_name, f'{ellipse_prefix}{appear_trans_prefix}{no_correct_prefix}_psi_{opt.psi}')
    print(output_dir)

    os.makedirs(output_dir, exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum = getattr(curriculums, curriculum_name)
    metadata = curriculums.extract_metadata(curriculum, 100e3)

    deformnet_ins = set_deformnet(metadata, device, opt)
    generator_core = getattr(siren, metadata['model'])
    generator = getattr(generators, metadata['generator'])(generator_core, metadata['latent_dim'],deformnet_ins = deformnet_ins, **metadata).to(device)

    generator.load_state_dict(torch.load(opt.generator_file, map_location=device))
    generator.set_device(device)
    generator.eval()

    if opt.ema:
        curriculum['ema'] = True
    if 'ema' in curriculum.keys() and curriculum['ema']:
        ema_file = opt.generator_file.split('generator')[0] + 'ema.pth'
        ema = torch.load(ema_file)
        ema.copy_to(generator.parameters())
    
    render_options = copy.deepcopy(metadata)
    render_options['img_size'] = opt.img_size
    render_options['num_steps'] = opt.num_steps
    render_options['stage'] = render_options['img_size']
    render_options['nerf_noise'] = 0
    render_options['alpha'] = 1.0
    render_options['h_stddev'] = 0.
    render_options['v_stddev'] = 0.
    render_options['psi'] = float(opt.psi)
    render_options['randomize'] = not metadata.get('sample_no_random', False)
    render_options['force_lastPoint_white'] = True

    if opt.render_template:
        generator.deformnet = None

    if not opt.ellipse_pose:
        pose_h_l = [a*np.pi*2 for a in np.linspace(0,1,opt.num_view)] 
        pose_v = metadata['v_mean']
        pose_v_h = [(pose_v, pose_h) for pose_h in pose_h_l]
    else:
        pose_v_h = get_ellipse_pose(metadata['h_mean'], metadata['v_mean'], metadata['h_stddev'], metadata['v_stddev'], opt.num_view)

    z2 = None
    if opt.input_appear_seed > 0:
        torch.manual_seed(opt.input_appear_seed)
        z2 =  torch.randn((1, 256), device=device)

    for seed in tqdm(opt.input_seed_list):
        images = []
        depths = []

        torch.manual_seed(seed)
        z = torch.randn((1, 256), device=device)
            
        if opt.save_raw:
            os.makedirs(os.path.join(output_dir, f'seed_{seed:04d}'), exist_ok=True)
        
        for pose_v, pose_h in pose_v_h:

            render_options['h_mean'] = pose_h #+ np.pi
            render_options['v_mean'] = pose_v

            with torch.no_grad():
                res = generator.staged_forward(z, no_correct=opt.no_correction, z2 = z2, render_normal=opt.render_normal, **render_options)
    
            tensor_img = res[0].detach().squeeze()
            tensor_depth = res[1].detach().squeeze()
            tensor_depth[tensor_depth!=tensor_depth] = render_options['ray_end']
            images.append(tensor_img)
            depths.append(tensor_depth)

        images = torch.stack(images)
        depths = torch.stack(depths)

        ## save gif image
        img = normalize_img(images)
        depth = normalize_depth(depths)
        depth_l = []
        img_l = []
        
        for i in range(img.shape[0]):
            pil_img = Image.fromarray(img[i]).convert("P",palette=Image.ADAPTIVE)
            pil_depth = Image.fromarray(depth[i]).convert("P",palette=Image.ADAPTIVE)
            
            if opt.save_raw:
                pil_img.save(os.path.join(output_dir, f'seed_{seed:04d}/view_{i:04d}.png'))
                pil_depth.save(os.path.join(output_dir, f'seed_{seed:04d}/depth_view_{i:04d}.png'))
            img_l.append(pil_img)
            depth_l.append(pil_depth)

        img_l[0].save(os.path.join(output_dir, f'seed_{seed:04d}.gif'),
            save_all = True, append_images = img_l[1:], 
            optimize = False, duration = 100, loop=0)

        depth_l[0].save(os.path.join(output_dir, f'seed_{seed:04d}_depth.gif'),
            save_all = True, append_images = depth_l[1:], 
            optimize = False, duration = 100, loop=0)


if __name__ == '__main__':
    main()