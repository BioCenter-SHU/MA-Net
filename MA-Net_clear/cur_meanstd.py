import torch
from data import dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time

path = "/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/" #我这里使用了绝对路径是因为我老是出错，搞得文件到处乱放
batch_size = 1
frames = 1
train_data = dataset.dataset_video(is_train=True, frame_path=path, root='frame_120/KFold_302_video.txt', clip_len=frames)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
def get_mean_std_value(loader, batch_size, frames):

    channels_sum,channel_squared_sum,num_batches = 0,0,0

    for data,target in loader:
        n, t, c, h, w = data.size()
        data = data.view(-1, 3, h, w)
        channels_sum += torch.mean(data,dim=[0,2,3])#shape [n_samples(batch),channels,height,width]
        #并不需要求channel的均值
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])#shape [n_samples(batch),channels,height,width]
        num_batches +=1
        print("video: {}".format(num_batches))

    # This lo calculate the summarized value of mean we need to divided it by num_batches

    mean = channels_sum/num_batches
    #这里将标准差 的公式变形了一下，让代码更方便写一点

    std = (channel_squared_sum/num_batches - mean**2)**5
    return mean,std

mean,std = get_mean_std_value(train_loader, batch_size, frames)
print('mean = {},std = {}'.format(mean,std))