import json
import os 
import sys
import pickle
import numpy as np
import pandas as pd
import random
import torch
import pdb
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import skimage.util as ski_util
from sklearn.utils import shuffle
import math
from copy import copy
import os
from re import L
from PIL import Image
from cv2 import CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE, _InputArray_STD_ARRAY
import torch
import torchvision
import sys
from torchvision.models import densenet161, resnet50, resnet101,resnet18
from PIL import Image
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from time import time
import time
import  csv
import pandas as pd
from models.spatial_transforms import *
from models.temporal_transforms import *
from opts import parser

class dataset_video(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, is_train, root, frame_path, clip_len):  # 初始化一些需要传入的参数
        super(dataset_video, self).__init__()

        rgb_samples = []
        labels = []

        fh = open(root, 'r',newline='')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        fh_reader = csv.reader(fh, delimiter='|')
        for line in fh_reader:  # 按行循环txt文本中的内容
            rgb_samples.append(line[0])
            labels.append(int(line[1]))
        print('{} videos have been loaded'.format(len(rgb_samples)))


        # 初始化
        self.rgb_samples = rgb_samples
        self.labels = labels
        self.is_train = is_train
        self.sample_num = len(self.rgb_samples)
        self.clip_len = clip_len
        self.frame_path = frame_path

        # 数据处理
        # input_mean=[.485, .456, .406]
        # input_std=[.229, .224, .225]
        input_mean = [0.1818, 0.1926, 0.2142]
        input_std = [.1922, .2026, .2236]
        normalize = GroupNormalize(input_mean, input_std)


        self.train_tsf = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(),
            ToTorchFormatTensor(),
            normalize   # z-score标准化
        ])
        self.temporal_transform = torchvision.transforms.Compose([
            TemporalUniformCrop_train(self.clip_len)
            ])

        self.val_tsf = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(),
            ToTorchFormatTensor(),
            normalize
        ])

        # self.train_tsf = torchvision.transforms.Compose([
        #     GroupScale([224, 224]),
        #     Stack(),
        #     ToTorchFormatTensor()
        # ])
        # self.temporal_transform = torchvision.transforms.Compose([
        #     TemporalUniformCrop_train(self.clip_len)
        #     ])

        # self.val_tsf = torchvision.transforms.Compose([
        #     GroupScale([224, 224]),
        #     Stack(),
        #     ToTorchFormatTensor()
        # ])

    def __getitem__(self, idx):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        rgb_name_str = self.rgb_samples[idx]
        rgb_name = rgb_name_str.strip(',').split(',')
        label = self.labels[idx]
        # frame文件名获取, txt中30个帧的list
        indices = [i for i in range(len(rgb_name))]
        # print('\nindices: {}'.format(indices))
        # indices: ['138/image_011.jpg', '138/image_019.jpg', '138/image_024.jpg', '138/image_030.jpg', '138/image_013.jpg', '138/image_009.jpg', '138/image_025.jpg', '138/image_022.jpg', '138/image_012.jpg', '138/image_003.jpg', '138/image_029.jpg', '138/image_008.jpg', '138/image_005.jpg', '138/image_028.jpg', '138/image_020.jpg', '138/image_016.jpg', '138/image_015.jpg', '138/image_021.jpg', '138/image_002.jpg', '138/image_023.jpg', '138/image_010.jpg', '138/image_006.jpg', '138/image_001.jpg', '138/image_018.jpg', '138/image_026.jpg', '138/image_004.jpg', '138/image_017.jpg', '138/image_007.jpg', '138/image_014.jpg', '138/image_027.jpg']
        
        if self.is_train:
            selected_indice = self.temporal_transform(indices)
        else:
            selected_indice = self.temporal_transform(indices)
        clip_rgb_frames = []
        clip_depth_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            # print('\nframe_name_i: {}'.format(rgb_name[frame_name_i]))
            rgb_cache = Image.open(os.path.join(self.frame_path, rgb_name[frame_name_i])).convert("RGB")
            clip_rgb_frames.append(rgb_cache)
        clip_rgb_frames = self.train_tsf(clip_rgb_frames)
        n, h, w = clip_rgb_frames.size()
        
        
        # return feature, label
        return clip_rgb_frames.view(-1, 3, h, w), label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return int(len(self.rgb_samples))



class dataset_video_inference(Dataset):
    def __init__(self, root_path, frame_path, clip_num = 2, clip_len = 16):

        rgb_samples = []
        labels = []

        fh = open(root_path, 'r',newline='')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        fh_reader = csv.reader(fh, delimiter='|')
        for line in fh_reader:  # 按行循环txt文本中的内容
            rgb_samples.append(line[0])
            labels.append(int(line[1]))
        print('{} videos have been loaded'.format(len(rgb_samples)))


        # 初始化
        self.clip_num = clip_num
        self.rgb_samples = rgb_samples
        self.labels = labels
        self.sample_num = len(self.rgb_samples)
        self.clip_len = clip_len
        self.frame_path = frame_path

        # 数据处理
        # input_mean=[.485, .456, .406]
        # input_std=[.229, .224, .225]
        # normalize = GroupNormalize(input_mean, input_std)
        input_mean = [0.1818, 0.1926, 0.2142]
        input_std = [.1922, .2026, .2236]
        normalize = GroupNormalize(input_mean, input_std)


        self.test_tsf = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(),
            ToTorchFormatTensor(),
            normalize   # z-score标准化
        ])
        self.temporal_transform = torchvision.transforms.Compose([
            TemporalUniformCrop_train(self.clip_len)
            ])

        # self.test_tsf = torchvision.transforms.Compose([
        #     GroupScale([224, 224]),
        #     Stack(),
        #     ToTorchFormatTensor(),
        #     normalize
        # ])
        # self.temporal_transform = torchvision.transforms.Compose([
        #     TemporalUniformCrop_train(self.clip_len)
        #     ])

    def __getitem__(self, idx):
        rgb_name_str = self.rgb_samples[idx]
        rgb_name = rgb_name_str.strip(',').split(',')
        label = self.labels[idx]
        # frame文件名获取, txt中30个帧的list
        indices = [i for i in range(len(rgb_name))]    
        video_clip = []
        for win_i in range(self.clip_num):
            clip_frames = []
            selected_indice = self.temporal_transform(copy(indices))
            for frame_name_i in selected_indice:
                rgb_cache = Image.open(os.path.join(self.frame_path , rgb_name[frame_name_i])).convert("RGB")
                clip_frames.append(rgb_cache)
            clip_frames = self.test_tsf(clip_frames)
            n, h, w = clip_frames.size()
            video_clip.append(clip_frames.view(-1, 3, h, w)) 
        video_clip = torch.stack(video_clip)
        return video_clip, int(label)

    def __len__(self):
        return int(self.sample_num)

