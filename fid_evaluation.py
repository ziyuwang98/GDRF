import os
import sys
import json
import argparse

import numpy as np
import torch

from tqdm import tqdm

from torchvision.utils import save_image
from PIL import Image

sys.path.append('./src')
from generators import generators
from siren import siren
import datasets
import curriculums
from training.model_setup import set_deformnet
from torch_fidelity import calculate_metrics

def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.png'), normalize=True, range=(-1, 1))
            img_counter += 1

def setup_evaluation(dataset_name, generated_dir, target_size=128, num_imgs=5000):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('FID_EVAL/EvalImages', dataset_name + '_real_images_' + str(target_size))
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, img_size=target_size)
        print('outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir)
        print('...done')

    os.makedirs(generated_dir, exist_ok=True)
    return real_dir

def normalize_img(tensor_img, img_range = (-1.,1.)):
    img = tensor_img.detach().permute(0, 2, 3, 1).squeeze() #(B, H, W, 3)
    img = (img - img_range[0])/(img_range[1]-img_range[0])
    img = (img.cpu().numpy()*255).astype(np.uint8)
    return img


def generate_fid_img(opt):
    curriculum_name = opt.curriculum
    metrics_path = os.path.join(opt.output_dir, curriculum_name, f'metric_numImg_{opt.num_images}_{opt.real_folder_name}.json')
    if os.path.exists(metrics_path):
        return

    output_dir = os.path.join(opt.output_dir, curriculum_name, f'fid_eval_numImg_{opt.num_images}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 50000)

    if opt.real_folder_name == '':
        real_images_dir = setup_evaluation(curriculum[0]['dataset'], opt.output_dir, target_size=metadata['img_size'])
        os.makedirs(opt.output_dir, exist_ok=True)
    else:
        real_images_dir = os.path.join('FID_EVAL/EvalImages', opt.real_folder_name)

    if not opt.only_eval:
        deformnet_ins = None
        if metadata.get('use_deformnet',False):
            deformnet_ins = set_deformnet(metadata, device, opt)
        generator_core = getattr(siren, metadata['model'])
        generator = getattr(generators, metadata['generator'])(generator_core, metadata['latent_dim'], deformnet_ins = deformnet_ins, **metadata).to(device)

        generator.load_state_dict(torch.load(opt.generator_file, map_location=device))
        generator.set_device(device)
        generator.eval()

        if opt.ema:
            curriculum['ema'] = True
        if curriculum['ema']:
            ema_file = opt.generator_file.split('generator')[0] + 'ema.pth'
            ema = torch.load(ema_file)
            ema.copy_to(generator.parameters())

        render_options = metadata
        render_options['num_steps'] = 96
        render_options['stage'] = render_options['img_size']
        render_options['nerf_noise'] = 0
        render_options['alpha'] = 1.0

        render_options['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
        render_options['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
        render_options['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
        render_options['psi'] = 1.0
        render_options['randomize'] = not metadata.get('sample_no_random', False)

        render_options['force_lastPoint_white'] = True

        for img_counter in tqdm(range(opt.num_images)):

            torch.manual_seed(img_counter)
            z = torch.randn(1, 256, device=device)

            with torch.no_grad():
                img = generator.staged_forward(z, **render_options)[0].detach().squeeze()
                img = normalize_img(img.unsqueeze(0))

                pil_img = Image.fromarray(img.squeeze()).convert("P",palette=Image.ADAPTIVE)
                pil_img.save(os.path.join(output_dir, f'{img_counter:0>5}.png'))

    return metrics_path, output_dir, real_images_dir

def calc_fid_number(metrics_path, output_dir, real_images_dir):
    if not os.path.exists(os.path.join(output_dir, '00999.png')):
        return
    metrics_dict = calculate_metrics(input1=output_dir, input2=real_images_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='FID_EVAL/GenerateImages')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu id')
    parser.add_argument('--curriculum', type=str, default='')
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--keep_percentage', type=float, default='1.0')
    parser.add_argument('--ema', default=True, type=bool)
    parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--only_eval', action='store_true', help='Only eval metric, not generate images')
    parser.add_argument('--real_folder_name', default='',type=str, help='The name of the real image folder')

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    metrics_path, output_dir, real_images_dir = generate_fid_img(opt)
    calc_fid_number(metrics_path, output_dir, real_images_dir)