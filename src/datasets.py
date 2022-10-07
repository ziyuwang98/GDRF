import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import glob
import PIL
import math
import numpy as np
import time
from scipy.io import loadmat
import json

import sys
sys.path.append('..')

def read_pose(name,flip=False):
    P = loadmat(name)['angle']
    P_x = -(P[0,0] - 0.1) + math.pi/2
    if not flip:
        P_y = P[0,1] + math.pi/2
    else:
        P_y = -P[0,1] + math.pi/2

    P = torch.tensor([P_x,P_y],dtype=torch.float32)

    return P

def read_pose_npy(name,flip=False):
    P = np.load(name)
    P_x = P[0] + 0.14
    if not flip:
        P_y = P[1]
    else:
        P_y = -P[1] + math.pi

    P = torch.tensor([P_x,P_y],dtype=torch.float32)
    return P

def transform_matrix_to_camera_pos(c2w,flip=False):
    """
    Get camera position with transform matrix

    :param c2w: camera to world transform matrix
    :return: camera position on spherical coord
    """

    c2w[[0,1,2]] = c2w[[1,2,0]]
    pos = c2w[:, -1].squeeze()
    radius = float(np.linalg.norm(pos))
    theta = float(np.arctan2(-pos[0], pos[2]))
    phi = float(np.arctan(-pos[1] / np.linalg.norm(pos[::2])))
    theta = theta + np.pi * 0.5
    phi = phi + np.pi * 0.5
    if flip:
        theta = -theta + math.pi
    P = torch.tensor([phi,theta],dtype=torch.float32)
    return P

class CAR128_3_VIEW(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        self.img_flip = False
        if 'img_flip' in kwargs and kwargs['img_flip'] == True:
            self.img_flip = True

        ## load all image_path and camera pose
        for i in range(10):
            try:
                img_l = []
                phi_theta = []

                dataset_path = 'datasets/128_SHAPRNET_CARS_3_view'
                with open(os.path.join(dataset_path, 'camera_param.json'), 'r') as fp:
                    meta = json.load(fp)

                for img_name, cam_meta in meta.items():
                    img_l.append(os.path.join(dataset_path, img_name))
                    phi_theta.append((cam_meta['phi'],cam_meta['theta']))

                phi_theta = torch.tensor(phi_theta, dtype=torch.float32)

                self.data = img_l
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.all_phi_theta = phi_theta
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        flip = False
        if self.img_flip and (torch.rand(1) < 0.5):
            X = F.hflip(X)
            flip = True

        if self.real_pose:
            P = self.all_phi_theta[index]
            if flip:
                P[1] = -P[1]
        else:
            P = 0
        return X, P


class CAR64_3_VIEW(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        self.img_flip = False
        if 'img_flip' in kwargs and kwargs['img_flip'] == True:
            self.img_flip = True

        ## load all image_path and camera pose
        for i in range(10):
            try:
                img_l = []
                phi_theta = []

                dataset_path = 'datasets/64_SHAPRNET_CARS_3_view'

                with open(os.path.join(dataset_path, 'camera_param.json'), 'r') as fp:
                    meta = json.load(fp)

                for img_name, cam_meta in meta.items():
                    img_l.append(os.path.join(dataset_path, img_name))
                    phi_theta.append((cam_meta['phi'],cam_meta['theta']))

                phi_theta = torch.tensor(phi_theta, dtype=torch.float32)

                self.data = img_l
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.all_phi_theta = phi_theta
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        flip = False
        if self.img_flip and (torch.rand(1) < 0.5):
            X = F.hflip(X)
            flip = True

        if self.real_pose:
            P = self.all_phi_theta[index]
            if flip:
                P[1] = -P[1]
        else:
            P = 0
        return X, P

class CHAIR128_1_VIEW(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        self.img_flip = False
        if 'img_flip' in kwargs and kwargs['img_flip'] == True:
            self.img_flip = True

        ## load all image_path and camera pose
        for i in range(10):
            try:
                img_l = []
                phi_theta = []

                dataset_path = 'datasets/128_PHOTOSHAPE_CHAIR_1_view'
                with open(os.path.join(dataset_path, 'camera_param.json'), 'r') as fp:
                    meta = json.load(fp)

                for img_name, cam_meta in meta.items():
                    img_l.append(os.path.join(dataset_path, img_name))
                    phi_theta.append((cam_meta['phi'],cam_meta['theta']))

                phi_theta = torch.tensor(phi_theta, dtype=torch.float32)

                self.data = img_l
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.all_phi_theta = phi_theta
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        flip = False
        if self.img_flip and (torch.rand(1) < 0.5):
            X = F.hflip(X)
            flip = True

        if self.real_pose:
            P = self.all_phi_theta[index]
            if flip:
                P[1] = -P[1]
        else:
            P = 0
        return X, P

class CHAIR64_1_VIEW(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        self.img_flip = False
        if 'img_flip' in kwargs and kwargs['img_flip'] == True:
            self.img_flip = True

        ## load all image_path and camera pose
        for i in range(10):
            try:
                img_l = []
                phi_theta = []

                dataset_path = 'datasets/64_PHOTOSHAPE_CHAIR_1_view'
                with open(os.path.join(dataset_path, 'camera_param.json'), 'r') as fp:
                    meta = json.load(fp)

                for img_name, cam_meta in meta.items():
                    img_l.append(os.path.join(dataset_path, img_name))
                    phi_theta.append((cam_meta['phi'],cam_meta['theta']))

                phi_theta = torch.tensor(phi_theta, dtype=torch.float32)

                self.data = img_l
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.all_phi_theta = phi_theta
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        flip = False
        if self.img_flip and (torch.rand(1) < 0.5):
            X = F.hflip(X)
            flip = True

        if self.real_pose:
            P = self.all_phi_theta[index]
            if flip:
                P[1] = -P[1]
        else:
            P = 0
        return X, P

class AIRPLANE64_3_VIEW(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        self.img_flip = False
        if 'img_flip' in kwargs and kwargs['img_flip'] == True:
            self.img_flip = True

        ## load all image_path and camera pose
        for i in range(10):
            try:
                img_l = []
                phi_theta = []

                dataset_path = 'datasets/64_SHAPRNET_AIRPLANE_3_view'
                with open(os.path.join(dataset_path, 'camera_param.json'), 'r') as fp:
                    meta = json.load(fp)

                for img_name, cam_meta in meta.items():
                    img_l.append(os.path.join(dataset_path, img_name))
                    phi_theta.append((cam_meta['phi'],cam_meta['theta']))

                phi_theta = torch.tensor(phi_theta, dtype=torch.float32)

                self.data = img_l
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.all_phi_theta = phi_theta
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        flip = False
        if self.img_flip and (torch.rand(1) < 0.5):
            X = F.hflip(X)
            flip = True

        if self.real_pose:
            P = self.all_phi_theta[index]
            if flip:
                P[1] = -P[1]
        else:
            P = 0
        return X, P

class AIRPLANE128_3_VIEW(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        self.img_flip = False
        if 'img_flip' in kwargs and kwargs['img_flip'] == True:
            self.img_flip = True

        ## load all image_path and camera pose
        for i in range(10):
            try:
                img_l = []
                phi_theta = []
                dataset_path = 'datasets/128_SHAPRNET_AIRPLANE_3_view'
                with open(os.path.join(dataset_path, 'camera_param.json'), 'r') as fp:
                    meta = json.load(fp)

                for img_name, cam_meta in meta.items():
                    img_l.append(os.path.join(dataset_path, img_name))
                    phi_theta.append((cam_meta['phi'],cam_meta['theta']))

                phi_theta = torch.tensor(phi_theta, dtype=torch.float32)

                self.data = img_l
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.all_phi_theta = phi_theta
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        flip = False
        if self.img_flip and (torch.rand(1) < 0.5):
            X = F.hflip(X)
            flip = True

        if self.real_pose:
            P = self.all_phi_theta[index]
            if flip:
                P[1] = -P[1]
        else:
            P = 0
        return X, P

class CARLA64(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        for i in range(10):
            try:
                self.data = glob.glob(os.path.join('datasets/carla/carla64','*.png'))
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.pose = [os.path.join('datasets/carla/carla_poses',f.split('/')[-1].replace('.png','_extrinsics.npy')) for f in self.data]
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        flip = (torch.rand(1) < 0.5)
        if flip:
            X = F.hflip(X)
        if self.real_pose:
            P = transform_matrix_to_camera_pos(np.load(self.pose[index]), flip=flip)
        else:
            P = 0

        return X, P

class CARLA128(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        for i in range(10):
            try:
                self.data = glob.glob(os.path.join('datasets/carla/carla128','*.png'))
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.pose = [os.path.join('datasets/carla/carla_poses',f.split('/')[-1].replace('.png','_extrinsics.npy')) for f in self.data]
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        flip = (torch.rand(1) < 0.5)
        if flip:
            X = F.hflip(X)
        if self.real_pose:
            P = transform_matrix_to_camera_pos(np.load(self.pose[index]), flip=flip)
        else:
            P = 0
        return X, P

def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed(_dataset, world_size, rank, batch_size, **kwargs):

    sampler = torch.utils.data.distributed.DistributedSampler(
        _dataset,
        num_replicas=world_size,
        rank=rank,
    )
    
    dataloader = MultiEpochsDataLoader(
        _dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, 3


## MultiEpochsDataLoader and _RepeatSampler are from https://github.com/rwightman/pytorch-image-models
'''
   Copyright 2019 Ross Wightman

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)