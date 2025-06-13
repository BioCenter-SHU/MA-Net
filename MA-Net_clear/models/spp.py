from re import X
from turtle import forward
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
import timm
import torchvision

class SPP(nn.Module):
    def __init__(self, base_model):
        super(SPP, self).__init__()
        self.in_channel = 256
        self.base_model = base_model
        # resnet50 = timm.create_model(self.base_model, pretrained=True)
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

         # ===========使用1×1卷积，添加通道压缩部分==================
        # self.layer1_1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        # self.layer2_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.layer3_1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        # self.layer4_1 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1)
        # self.conv_cat = nn.Sequential(nn.Conv2d(256*4, 1024, kernel_size=3, stride=1, padding=1),
        #                     nn.ReLU(inplace=True),
        #                     nn.BatchNorm2d(1024))
        
        # ===========添加代码的末尾=================================

        self.avgpool = resnet50.avgpool
        self.fc = resnet50.fc
        self.cat_fc = nn.Linear(self.in_channel, 3)


    def forward(self, x):
        x_conv = self.conv1(x)
        x = self.bn1(x_conv)
        x = self.relu(x)
        x = self.maxpool(x) # 60, 64, 112, 112
        x1 = self.layer1(x) # 60, 256, 56, 56
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)


        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def make_spp(base_model):
    return SPP(base_model)