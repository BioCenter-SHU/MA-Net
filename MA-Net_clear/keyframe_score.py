import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from models import models as TSN_model
from torchvision import transforms
from gram_utils import GradCAM, show_cam_on_image, center_crop_img


from models.temporal_transforms import *
from models.spatial_transforms import *
import torchvision
from torch import nn
import cv2
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from data import dataset_jester, dataset_EgoGesture, dataset_sthv2, dataset

device = 'cpu' 
def main():
    # 建立模型
    # os.environ['CUDA_VISIBLE_DEVICES']='1'  
    model = TSN_model.TSN(3, 30, 'RGB', 
                                module='res4',  
                                base_model= "resnet50", 
                                shift_div = 8, 
                                dropout= 0.9, 
                                img_feature_dim = 224,
                                pretrain= True, # 'imagenet' or False
                                consensus_type='avg',
                                fc_lr5 = True,
                                print_spec = True)
    # 目标层
    # target_layers = model.base_model.layer3[5].conv1.keyframe_softamx[1][2]
    # 加载预训练权重
    pretrained_dict = torch.load('/YOUR_PATH/Clssification/DuoChiDu/checkpoint/res/res4/resnet50_30/2022-12-16-06-50-54/resnet50_30/clip_len_30_4_checkpoint.pth.tar', map_location='cpu')
    model.load_state_dict(pretrained_dict['state_dict'])
    # model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)

    val_dataloader = DataLoader(dataset.dataset_video(is_train=False,root="/YOUR_PATH/Clssification/DuoChiDu/annotation/kf_data.txt", 
                                                                    frame_path="/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/", 
                                                                    clip_len=30),
                                                                    batch_size=1, shuffle=True, num_workers=1
                                )
    inference(0, model, val_dataloader)
    
   
def inference(k, model, val_dataloader):

    model.eval()

    with torch.no_grad():
        for step, inputs in enumerate(tqdm(val_dataloader)):
            # pdb.set_trace()
            rgb, labels = inputs[0], inputs[1]
            rgb = rgb.to(device, non_blocking=True).float()
            outputs = model(rgb)
            outputs = outputs.view(1, 30, -1)
            outputs = F.softmax(outputs, 2)
            print(outputs.data)
            print(outputs.data.mean(1))

    

if __name__ == '__main__':
    main()
