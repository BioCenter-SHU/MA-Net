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

def main(frame_path, video_path, video_name, clip):
    # 建立模型
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = 'cuda:0'   
    model = TSN_model.TSN(3, clip, 'RGB', 
                                base_model= "resnet50", 
                                shift_div = 8, 
                                dropout= 0.9, 
                                img_feature_dim = 224,
                                pretrain= True, # 'imagenet' or False
                                consensus_type='avg',
                                fc_lr5 = True,
                                print_spec = True)
    # 目标层
    target_layers = [model.base_model.layer4]
    # target_layers = [model.new_fc]
    # 加载预训练权重
    pretrained_dict = torch.load('checkpoint/baseline/resnet50_30/2022-09-14-23-16-48/resnet50_30/clip_len_30_0_checkpoint.pth.tar', map_location='cpu')
    model.load_state_dict(pretrained_dict['state_dict'])
    model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)
    # 数据预处理
    data_transform = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(),
            ToTorchFormatTensor(),
            # mean和std是我数据集用的，需要修改成你自己的
            GroupNormalize([0.1818, 0.1926, 0.2142], [.1922, .2026, .2236])
        ])
    temporal_transform = torchvision.transforms.Compose([
            TemporalUniformCrop_train(clip)
            ])


    # load video
    rgb_name = video_path.strip(',').split(',')
    # frame文件名获取, txt中30个帧的list
    indices = [i for i in range(len(rgb_name))]
    # 选取帧的index存入selected_indice
    selected_indice = temporal_transform(indices)
    # 读取frames
    clip_rgb_frames = []
    for i, frame_name_i in enumerate(selected_indice):
        rgb_cache = Image.open(os.path.join(frame_path, rgb_name[frame_name_i])).convert("RGB") # 740,540
        clip_rgb_frames.append(rgb_cache)
    video_frames = data_transform(clip_rgb_frames)   # 90, 224, 224
    n, h, w = video_frames.size()
    # 这里clip要修改成自己的
    video_clip = video_frames.view(-1, clip, 3, h, w)  # n, t, c, h, w
    # print(video_clip)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # 类别
    target_category = 2  # tabby, tabby cat

    grayscale_cam = cam(input_tensor=video_clip, target_category=target_category)   # 30, 224, 224

    # 设置保存路径
    save_path = 'visualization/resnet50/{}'.format(video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(clip):
        grayscale_cam_i = grayscale_cam[i, :] # 把第i张cam提取出来
        img = clip_rgb_frames[i]    # 第i帧
        img = np.array(img, dtype=np.uint8) 
        img = cv2.resize(img, (224, 224))
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                        grayscale_cam_i,
                                        use_rgb=True)
        # 改为自己数据集的分辨率
        pre = cv2.resize(visualization, (740,540))
        plt.imshow(pre)
        plt.show()
        plt.savefig(os.path.join(save_path, 'pre_{}.png'.format(selected_indice[i])))
    


if __name__ == '__main__':
    frame_path = "/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/"
    video_path = 'Mixed/233/0.jpg,Mixed/233/1.jpg,Mixed/233/2.jpg,Mixed/233/3.jpg,Mixed/233/4.jpg,Mixed/233/5.jpg,Mixed/233/6.jpg,Mixed/233/7.jpg,Mixed/233/8.jpg,Mixed/233/9.jpg,Mixed/233/10.jpg,Mixed/233/11.jpg,Mixed/233/12.jpg,Mixed/233/13.jpg,Mixed/233/14.jpg,Mixed/233/15.jpg,Mixed/233/16.jpg,Mixed/233/17.jpg,Mixed/233/18.jpg,Mixed/233/19.jpg,Mixed/233/20.jpg,Mixed/233/21.jpg,Mixed/233/22.jpg,Mixed/233/23.jpg,Mixed/233/24.jpg,Mixed/233/25.jpg,Mixed/233/26.jpg,Mixed/233/27.jpg,Mixed/233/28.jpg,Mixed/233/29.jpg,Mixed/233/30.jpg,Mixed/233/31.jpg,Mixed/233/32.jpg,Mixed/233/33.jpg,Mixed/233/34.jpg,Mixed/233/35.jpg,Mixed/233/36.jpg,Mixed/233/37.jpg,Mixed/233/38.jpg,Mixed/233/39.jpg,Mixed/233/40.jpg,Mixed/233/41.jpg,Mixed/233/42.jpg,Mixed/233/43.jpg,Mixed/233/44.jpg,Mixed/233/45.jpg,Mixed/233/46.jpg,Mixed/233/47.jpg,Mixed/233/48.jpg,Mixed/233/49.jpg,Mixed/233/50.jpg,Mixed/233/51.jpg,Mixed/233/52.jpg,Mixed/233/53.jpg,Mixed/233/54.jpg,Mixed/233/55.jpg,Mixed/233/56.jpg,Mixed/233/57.jpg,Mixed/233/58.jpg,Mixed/233/59.jpg,Mixed/233/60.jpg,Mixed/233/61.jpg,Mixed/233/62.jpg,Mixed/233/63.jpg,Mixed/233/64.jpg,Mixed/233/65.jpg,Mixed/233/66.jpg,Mixed/233/67.jpg,Mixed/233/68.jpg,Mixed/233/69.jpg,Mixed/233/70.jpg,Mixed/233/71.jpg,Mixed/233/72.jpg,Mixed/233/73.jpg,Mixed/233/74.jpg,Mixed/233/75.jpg,Mixed/233/76.jpg,Mixed/233/77.jpg,Mixed/233/78.jpg,Mixed/233/79.jpg,Mixed/233/80.jpg,Mixed/233/81.jpg,Mixed/233/82.jpg,Mixed/233/83.jpg,Mixed/233/84.jpg,Mixed/233/85.jpg,Mixed/233/86.jpg,Mixed/233/87.jpg,Mixed/233/88.jpg,Mixed/233/89.jpg,Mixed/233/90.jpg,Mixed/233/91.jpg,Mixed/233/92.jpg,Mixed/233/93.jpg,Mixed/233/94.jpg,Mixed/233/95.jpg,Mixed/233/96.jpg,Mixed/233/97.jpg,Mixed/233/98.jpg,Mixed/233/99.jpg,Mixed/233/100.jpg,Mixed/233/101.jpg,Mixed/233/102.jpg,Mixed/233/103.jpg,Mixed/233/104.jpg,Mixed/233/105.jpg,Mixed/233/106.jpg,Mixed/233/107.jpg,Mixed/233/108.jpg,Mixed/233/109.jpg,Mixed/233/110.jpg,Mixed/233/111.jpg,Mixed/233/112.jpg,Mixed/233/113.jpg,Mixed/233/114.jpg,Mixed/233/115.jpg,Mixed/233/116.jpg,Mixed/233/117.jpg,Mixed/233/118.jpg,Mixed/233/119.jpg'
    video_name = 'Mixed/233'
    # video_path = 'Mixed/63/0.jpg,Mixed/63/1.jpg,Mixed/63/2.jpg,Mixed/63/3.jpg,Mixed/63/4.jpg,Mixed/63/5.jpg,Mixed/63/6.jpg,Mixed/63/7.jpg,Mixed/63/8.jpg,Mixed/63/9.jpg,Mixed/63/10.jpg,Mixed/63/11.jpg,Mixed/63/12.jpg,Mixed/63/13.jpg,Mixed/63/14.jpg,Mixed/63/15.jpg,Mixed/63/16.jpg,Mixed/63/17.jpg,Mixed/63/18.jpg,Mixed/63/19.jpg,Mixed/63/20.jpg,Mixed/63/21.jpg,Mixed/63/22.jpg,Mixed/63/23.jpg,Mixed/63/24.jpg,Mixed/63/25.jpg,Mixed/63/26.jpg,Mixed/63/27.jpg,Mixed/63/28.jpg,Mixed/63/29.jpg,Mixed/63/30.jpg,Mixed/63/31.jpg,Mixed/63/32.jpg,Mixed/63/33.jpg,Mixed/63/34.jpg,Mixed/63/35.jpg,Mixed/63/36.jpg,Mixed/63/37.jpg,Mixed/63/38.jpg,Mixed/63/39.jpg,Mixed/63/40.jpg,Mixed/63/41.jpg,Mixed/63/42.jpg,Mixed/63/43.jpg,Mixed/63/44.jpg,Mixed/63/45.jpg,Mixed/63/46.jpg,Mixed/63/47.jpg,Mixed/63/48.jpg,Mixed/63/49.jpg,Mixed/63/50.jpg,Mixed/63/51.jpg,Mixed/63/52.jpg,Mixed/63/53.jpg,Mixed/63/54.jpg,Mixed/63/55.jpg,Mixed/63/56.jpg,Mixed/63/57.jpg,Mixed/63/58.jpg,Mixed/63/59.jpg,Mixed/63/60.jpg,Mixed/63/61.jpg,Mixed/63/62.jpg,Mixed/63/63.jpg,Mixed/63/64.jpg,Mixed/63/65.jpg,Mixed/63/66.jpg,Mixed/63/67.jpg,Mixed/63/68.jpg,Mixed/63/69.jpg,Mixed/63/70.jpg,Mixed/63/71.jpg,Mixed/63/72.jpg,Mixed/63/73.jpg,Mixed/63/74.jpg,Mixed/63/75.jpg,Mixed/63/76.jpg,Mixed/63/77.jpg,Mixed/63/78.jpg,Mixed/63/79.jpg,Mixed/63/80.jpg,Mixed/63/81.jpg,Mixed/63/82.jpg,Mixed/63/83.jpg,Mixed/63/84.jpg,Mixed/63/85.jpg,Mixed/63/86.jpg,Mixed/63/87.jpg,Mixed/63/88.jpg,Mixed/63/89.jpg,Mixed/63/90.jpg,Mixed/63/91.jpg,Mixed/63/92.jpg,Mixed/63/93.jpg,Mixed/63/94.jpg,Mixed/63/95.jpg,Mixed/63/96.jpg,Mixed/63/97.jpg,Mixed/63/98.jpg,Mixed/63/99.jpg,Mixed/63/100.jpg,Mixed/63/101.jpg,Mixed/63/102.jpg,Mixed/63/103.jpg,Mixed/63/104.jpg,Mixed/63/105.jpg,Mixed/63/106.jpg,Mixed/63/107.jpg,Mixed/63/108.jpg,Mixed/63/109.jpg,Mixed/63/110.jpg,Mixed/63/111.jpg,Mixed/63/112.jpg,Mixed/63/113.jpg,Mixed/63/114.jpg,Mixed/63/115.jpg,Mixed/63/116.jpg,Mixed/63/117.jpg,Mixed/63/118.jpg,Mixed/63/119.jpg'
    # video_name = 'Mixed/63'
    # video_path = 'Stable/81/0.jpg,Stable/81/1.jpg,Stable/81/2.jpg,Stable/81/3.jpg,Stable/81/4.jpg,Stable/81/5.jpg,Stable/81/6.jpg,Stable/81/7.jpg,Stable/81/8.jpg,Stable/81/9.jpg,Stable/81/10.jpg,Stable/81/11.jpg,Stable/81/12.jpg,Stable/81/13.jpg,Stable/81/14.jpg,Stable/81/15.jpg,Stable/81/16.jpg,Stable/81/17.jpg,Stable/81/18.jpg,Stable/81/19.jpg,Stable/81/20.jpg,Stable/81/21.jpg,Stable/81/22.jpg,Stable/81/23.jpg,Stable/81/24.jpg,Stable/81/25.jpg,Stable/81/26.jpg,Stable/81/27.jpg,Stable/81/28.jpg,Stable/81/29.jpg,Stable/81/30.jpg,Stable/81/31.jpg,Stable/81/32.jpg,Stable/81/33.jpg,Stable/81/34.jpg,Stable/81/35.jpg,Stable/81/36.jpg,Stable/81/37.jpg,Stable/81/38.jpg,Stable/81/39.jpg,Stable/81/40.jpg,Stable/81/41.jpg,Stable/81/42.jpg,Stable/81/43.jpg,Stable/81/44.jpg,Stable/81/45.jpg,Stable/81/46.jpg,Stable/81/47.jpg,Stable/81/48.jpg,Stable/81/49.jpg,Stable/81/50.jpg,Stable/81/51.jpg,Stable/81/52.jpg,Stable/81/53.jpg,Stable/81/54.jpg,Stable/81/55.jpg,Stable/81/56.jpg,Stable/81/57.jpg,Stable/81/58.jpg,Stable/81/59.jpg,Stable/81/60.jpg,Stable/81/61.jpg,Stable/81/62.jpg,Stable/81/63.jpg,Stable/81/64.jpg,Stable/81/65.jpg,Stable/81/66.jpg,Stable/81/67.jpg,Stable/81/68.jpg,Stable/81/69.jpg,Stable/81/70.jpg,Stable/81/71.jpg,Stable/81/72.jpg,Stable/81/73.jpg,Stable/81/74.jpg,Stable/81/75.jpg,Stable/81/76.jpg,Stable/81/77.jpg,Stable/81/78.jpg,Stable/81/79.jpg,Stable/81/80.jpg,Stable/81/81.jpg,Stable/81/82.jpg,Stable/81/83.jpg,Stable/81/84.jpg,Stable/81/85.jpg,Stable/81/86.jpg,Stable/81/87.jpg,Stable/81/88.jpg,Stable/81/89.jpg,Stable/81/90.jpg,Stable/81/91.jpg,Stable/81/92.jpg,Stable/81/93.jpg,Stable/81/94.jpg,Stable/81/95.jpg,Stable/81/96.jpg,Stable/81/97.jpg,Stable/81/98.jpg,Stable/81/99.jpg,Stable/81/100.jpg,Stable/81/101.jpg,Stable/81/102.jpg,Stable/81/103.jpg,Stable/81/104.jpg,Stable/81/105.jpg,Stable/81/106.jpg,Stable/81/107.jpg,Stable/81/108.jpg,Stable/81/109.jpg,Stable/81/110.jpg,Stable/81/111.jpg,Stable/81/112.jpg,Stable/81/113.jpg,Stable/81/114.jpg,Stable/81/115.jpg,Stable/81/116.jpg,Stable/81/117.jpg,Stable/81/118.jpg,Stable/81/119.jpg'
    # video_name = 'Stable/81'
    # video_path = 'Stable/226/0.jpg,Stable/226/1.jpg,Stable/226/2.jpg,Stable/226/3.jpg,Stable/226/4.jpg,Stable/226/5.jpg,Stable/226/6.jpg,Stable/226/7.jpg,Stable/226/8.jpg,Stable/226/9.jpg,Stable/226/10.jpg,Stable/226/11.jpg,Stable/226/12.jpg,Stable/226/13.jpg,Stable/226/14.jpg,Stable/226/15.jpg,Stable/226/16.jpg,Stable/226/17.jpg,Stable/226/18.jpg,Stable/226/19.jpg,Stable/226/20.jpg,Stable/226/21.jpg,Stable/226/22.jpg,Stable/226/23.jpg,Stable/226/24.jpg,Stable/226/25.jpg,Stable/226/26.jpg,Stable/226/27.jpg,Stable/226/28.jpg,Stable/226/29.jpg,Stable/226/30.jpg,Stable/226/31.jpg,Stable/226/32.jpg,Stable/226/33.jpg,Stable/226/34.jpg,Stable/226/35.jpg,Stable/226/36.jpg,Stable/226/37.jpg,Stable/226/38.jpg,Stable/226/39.jpg,Stable/226/40.jpg,Stable/226/41.jpg,Stable/226/42.jpg,Stable/226/43.jpg,Stable/226/44.jpg,Stable/226/45.jpg,Stable/226/46.jpg,Stable/226/47.jpg,Stable/226/48.jpg,Stable/226/49.jpg,Stable/226/50.jpg,Stable/226/51.jpg,Stable/226/52.jpg,Stable/226/53.jpg,Stable/226/54.jpg,Stable/226/55.jpg,Stable/226/56.jpg,Stable/226/57.jpg,Stable/226/58.jpg,Stable/226/59.jpg,Stable/226/60.jpg,Stable/226/61.jpg,Stable/226/62.jpg,Stable/226/63.jpg,Stable/226/64.jpg,Stable/226/65.jpg,Stable/226/66.jpg,Stable/226/67.jpg,Stable/226/68.jpg,Stable/226/69.jpg,Stable/226/70.jpg,Stable/226/71.jpg,Stable/226/72.jpg,Stable/226/73.jpg,Stable/226/74.jpg,Stable/226/75.jpg,Stable/226/76.jpg,Stable/226/77.jpg,Stable/226/78.jpg,Stable/226/79.jpg,Stable/226/80.jpg,Stable/226/81.jpg,Stable/226/82.jpg,Stable/226/83.jpg,Stable/226/84.jpg,Stable/226/85.jpg,Stable/226/86.jpg,Stable/226/87.jpg,Stable/226/88.jpg,Stable/226/89.jpg,Stable/226/90.jpg,Stable/226/91.jpg,Stable/226/92.jpg,Stable/226/93.jpg,Stable/226/94.jpg,Stable/226/95.jpg,Stable/226/96.jpg,Stable/226/97.jpg,Stable/226/98.jpg,Stable/226/99.jpg,Stable/226/100.jpg,Stable/226/101.jpg,Stable/226/102.jpg,Stable/226/103.jpg,Stable/226/104.jpg,Stable/226/105.jpg,Stable/226/106.jpg,Stable/226/107.jpg,Stable/226/108.jpg,Stable/226/109.jpg,Stable/226/110.jpg,Stable/226/111.jpg,Stable/226/112.jpg,Stable/226/113.jpg,Stable/226/114.jpg,Stable/226/115.jpg,Stable/226/116.jpg,Stable/226/117.jpg,Stable/226/118.jpg,Stable/226/119.jpg'
    # video_name = 'Stable/226'
    # video_path = 'Stable/134/0.jpg,Stable/134/1.jpg,Stable/134/2.jpg,Stable/134/3.jpg,Stable/134/4.jpg,Stable/134/5.jpg,Stable/134/6.jpg,Stable/134/7.jpg,Stable/134/8.jpg,Stable/134/9.jpg,Stable/134/10.jpg,Stable/134/11.jpg,Stable/134/12.jpg,Stable/134/13.jpg,Stable/134/14.jpg,Stable/134/15.jpg,Stable/134/16.jpg,Stable/134/17.jpg,Stable/134/18.jpg,Stable/134/19.jpg,Stable/134/20.jpg,Stable/134/21.jpg,Stable/134/22.jpg,Stable/134/23.jpg,Stable/134/24.jpg,Stable/134/25.jpg,Stable/134/26.jpg,Stable/134/27.jpg,Stable/134/28.jpg,Stable/134/29.jpg,Stable/134/30.jpg,Stable/134/31.jpg,Stable/134/32.jpg,Stable/134/33.jpg,Stable/134/34.jpg,Stable/134/35.jpg,Stable/134/36.jpg,Stable/134/37.jpg,Stable/134/38.jpg,Stable/134/39.jpg,Stable/134/40.jpg,Stable/134/41.jpg,Stable/134/42.jpg,Stable/134/43.jpg,Stable/134/44.jpg,Stable/134/45.jpg,Stable/134/46.jpg,Stable/134/47.jpg,Stable/134/48.jpg,Stable/134/49.jpg,Stable/134/50.jpg,Stable/134/51.jpg,Stable/134/52.jpg,Stable/134/53.jpg,Stable/134/54.jpg,Stable/134/55.jpg,Stable/134/56.jpg,Stable/134/57.jpg,Stable/134/58.jpg,Stable/134/59.jpg,Stable/134/60.jpg,Stable/134/61.jpg,Stable/134/62.jpg,Stable/134/63.jpg,Stable/134/64.jpg,Stable/134/65.jpg,Stable/134/66.jpg,Stable/134/67.jpg,Stable/134/68.jpg,Stable/134/69.jpg,Stable/134/70.jpg,Stable/134/71.jpg,Stable/134/72.jpg,Stable/134/73.jpg,Stable/134/74.jpg,Stable/134/75.jpg,Stable/134/76.jpg,Stable/134/77.jpg,Stable/134/78.jpg,Stable/134/79.jpg,Stable/134/80.jpg,Stable/134/81.jpg,Stable/134/82.jpg,Stable/134/83.jpg,Stable/134/84.jpg,Stable/134/85.jpg,Stable/134/86.jpg,Stable/134/87.jpg,Stable/134/88.jpg,Stable/134/89.jpg,Stable/134/90.jpg,Stable/134/91.jpg,Stable/134/92.jpg,Stable/134/93.jpg,Stable/134/94.jpg,Stable/134/95.jpg,Stable/134/96.jpg,Stable/134/97.jpg,Stable/134/98.jpg,Stable/134/99.jpg,Stable/134/100.jpg,Stable/134/101.jpg,Stable/134/102.jpg,Stable/134/103.jpg,Stable/134/104.jpg,Stable/134/105.jpg,Stable/134/106.jpg,Stable/134/107.jpg,Stable/134/108.jpg,Stable/134/109.jpg,Stable/134/110.jpg,Stable/134/111.jpg,Stable/134/112.jpg,Stable/134/113.jpg,Stable/134/114.jpg,Stable/134/115.jpg,Stable/134/116.jpg,Stable/134/117.jpg,Stable/134/118.jpg,Stable/134/119.jpg'
    # video_name = 'Stable/134'
    # video_path = 'Unstable/34/0.jpg,Unstable/34/1.jpg,Unstable/34/2.jpg,Unstable/34/3.jpg,Unstable/34/4.jpg,Unstable/34/5.jpg,Unstable/34/6.jpg,Unstable/34/7.jpg,Unstable/34/8.jpg,Unstable/34/9.jpg,Unstable/34/10.jpg,Unstable/34/11.jpg,Unstable/34/12.jpg,Unstable/34/13.jpg,Unstable/34/14.jpg,Unstable/34/15.jpg,Unstable/34/16.jpg,Unstable/34/17.jpg,Unstable/34/18.jpg,Unstable/34/19.jpg,Unstable/34/20.jpg,Unstable/34/21.jpg,Unstable/34/22.jpg,Unstable/34/23.jpg,Unstable/34/24.jpg,Unstable/34/25.jpg,Unstable/34/26.jpg,Unstable/34/27.jpg,Unstable/34/28.jpg,Unstable/34/29.jpg,Unstable/34/30.jpg,Unstable/34/31.jpg,Unstable/34/32.jpg,Unstable/34/33.jpg,Unstable/34/34.jpg,Unstable/34/35.jpg,Unstable/34/36.jpg,Unstable/34/37.jpg,Unstable/34/38.jpg,Unstable/34/39.jpg,Unstable/34/40.jpg,Unstable/34/41.jpg,Unstable/34/42.jpg,Unstable/34/43.jpg,Unstable/34/44.jpg,Unstable/34/45.jpg,Unstable/34/46.jpg,Unstable/34/47.jpg,Unstable/34/48.jpg,Unstable/34/49.jpg,Unstable/34/50.jpg,Unstable/34/51.jpg,Unstable/34/52.jpg,Unstable/34/53.jpg,Unstable/34/54.jpg,Unstable/34/55.jpg,Unstable/34/56.jpg,Unstable/34/57.jpg,Unstable/34/58.jpg,Unstable/34/59.jpg,Unstable/34/60.jpg,Unstable/34/61.jpg,Unstable/34/62.jpg,Unstable/34/63.jpg,Unstable/34/64.jpg,Unstable/34/65.jpg,Unstable/34/66.jpg,Unstable/34/67.jpg,Unstable/34/68.jpg,Unstable/34/69.jpg,Unstable/34/70.jpg,Unstable/34/71.jpg,Unstable/34/72.jpg,Unstable/34/73.jpg,Unstable/34/74.jpg,Unstable/34/75.jpg,Unstable/34/76.jpg,Unstable/34/77.jpg,Unstable/34/78.jpg,Unstable/34/79.jpg,Unstable/34/80.jpg,Unstable/34/81.jpg,Unstable/34/82.jpg,Unstable/34/83.jpg,Unstable/34/84.jpg,Unstable/34/85.jpg,Unstable/34/86.jpg,Unstable/34/87.jpg,Unstable/34/88.jpg,Unstable/34/89.jpg,Unstable/34/90.jpg,Unstable/34/91.jpg,Unstable/34/92.jpg,Unstable/34/93.jpg,Unstable/34/94.jpg,Unstable/34/95.jpg,Unstable/34/96.jpg,Unstable/34/97.jpg,Unstable/34/98.jpg,Unstable/34/99.jpg,Unstable/34/100.jpg,Unstable/34/101.jpg,Unstable/34/102.jpg,Unstable/34/103.jpg,Unstable/34/104.jpg,Unstable/34/105.jpg,Unstable/34/106.jpg,Unstable/34/107.jpg,Unstable/34/108.jpg,Unstable/34/109.jpg,Unstable/34/110.jpg,Unstable/34/111.jpg,Unstable/34/112.jpg,Unstable/34/113.jpg,Unstable/34/114.jpg,Unstable/34/115.jpg,Unstable/34/116.jpg,Unstable/34/117.jpg,Unstable/34/118.jpg,Unstable/34/119.jpg'
    # video_name = 'Unstable/34'
    main(frame_path, video_path, video_name, 30)
