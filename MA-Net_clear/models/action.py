from turtle import forward
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from models.spp import SPP


class CBAMBlock(nn.Module):
    def __init__(self):
        super(CBAMBlock).__init__()
        # self.net = net
        # self.n_segment = n_segment
        # self.in_channels = self.net.in_channels
        # self.out_channels = self.net.out_channels
        # self.kernel_size = self.net.kernel_size
        # self.stride = self.net.stride
        # self.padding = self.net.padding
        # self.reduced_channels = self.in_channels//16
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        # self.fold = self.in_channels // shift_div
        # self.softmax = nn.Softmax(dim=1)
        # self.leackrelu = nn.LeakyReLU(inplace=True)
        # self.cat_conv2d = nn.Conv2d(2 * self.in_channels, self.in_channels, 3, 1, 1)
    
    def forward(self, x):
        # # ======================================通道注意力 （2）=========================================
        # # sigmoid前avg+max
        # x_p2_1 = self.avg_pool(x)
        # x_p2_1 = self.action_p2_squeeze(x_p2_1)
        # nt, c, h, w = x_p2_1.size()
        # x_p2_1 = x_p2_1.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        # x_p2_1 = self.action_p2_conv1(x_p2_1)
        # x_p2_1 = self.relu(x_p2_1)
        # x_p2_1 = x_p2_1.transpose(2,1).contiguous().view(-1, c, 1, 1)
        # x_p2_1 = self.action_p2_expand(x_p2_1)

        # x_p2_2 = self.max_pool(x)
        # x_p2_2 = self.action_p2_squeeze(x_p2_2)
        # nt, c, h, w = x_p2_2.size()
        # x_p2_2 = x_p2_2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        # x_p2_2 = self.action_p2_conv1(x_p2_2)
        # x_p2_2 = self.relu(x_p2_2)
        # x_p2_2 = x_p2_2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        # x_p2_2 = self.action_p2_expand(x_p2_2)
        # x_p2 = self.sigmoid(x_p2_1 + x_p2_2)
        # x_p2 = x * x_p2 + x # nt, c, h, w


        #===============================时空注意力======================================
        # # reshape
        # x_p1 = x_p2.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous() # n_batch, c, T, h, w
        # # # 平均
        # x_p1_avg = x_p1.mean(1, keepdim=True)   # n_batch, 1, t, h, w
        # x_p1_max = x_p1.max(1, keepdim=True)    # n_batch, 1, t, h, w
        # x_p1 = torch.cat([x_p1_avg, x_p1_max], dim=1)   # n_batch, 2, t, h, w
        # x_p1 = self.action_p1_conv1(x_p1)   # n_batch, 1, t, h, w
        # # reshape
        # x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)   # nt, 1, h, w
        # x_p1 = self.sigmoid(x_p1)
        # x_p1 = x_p2 * x_p1 + x_p2
        return 

class Action(nn.Module):
    def __init__(self, net, n_segment=3, shift_div=8):
        super(Action, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.in_channels = self.net.in_channels
        self.out_channels = self.net.out_channels
        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = self.net.padding
        self.reduced_channels = self.in_channels//16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div
        self.softmax = nn.Softmax(dim=2)
        self.leackrelu = nn.LeakyReLU(inplace=True)
        self.cat_conv2d = nn.Conv2d(2 * self.in_channels, self.in_channels, 3, 1, 1)

        # Background Attenuation
        # self.background_att = nn.Conv2d()

        # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(2, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))  

        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
    

        print('====USE CBAMBlock====')
        # motion excitation
        # self.pad = (0,0,0,0,0,0,0,1)
        # self.pad = (0,0,0,0,0,0,1,0)
        # self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        # self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        # # self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
        # #                             stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        # self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        # print('=> Using motion')
        

        


    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        # x_p1 = x_shift.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous()
        # x_p1 = x_p1.mean(1, keepdim=True)
        # x_p1 = self.action_p1_conv1(x_p1)
        # x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        # x_p1 = self.softmax(x_p1)
        # # x_p1 = self.sigmoid(x_p1)
        # x_p1 = x_shift * x_p1 + x_shift


        # 多尺度注意力
        # x_p1 = x.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous()
        # x_p1 = x_p1.mean(1, keepdim=True)
        # x_p1_1 = self.action_p11_conv1(x_p1)
        # x_p1_2 = self.action_p12_conv1(x_p1)
        # x_p1_3 = self.action_p13_conv1(x_p1)
        # x_p1_4 = self.action_p14_conv1(x_p1)
        # x_p1 = x_p1_1 + x_p1_2 + x_p1_3 + x_p1_4
        # x_p1_1 = self.conv_11(x_p1)
        # x_p1_2 = self.conv_12(x_p1)
        # x_p1_3 = self.conv_13(x_p1)
        # x_p1_4 = self.conv_14(x_p1)
        # x_p1 = torch.cat((x_p1_1, x_p1_2, x_p1_3, x_p1_4),dim=1)
        # x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        # x_p1 = self.softmax(x_p1)
        # x_p1 = self.sigmoid(x_p1)
        # x_p1 = x * x_p1 + x


        # 3D convolution: c*T*h*w, spatial temporal excitation
        # x_p1 = x.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous()
        # # # 平均
        # x_p1 = x_p1.mean(1, keepdim=True)
        # x_p1 = self.action_p1_conv1(x_p1)
        # # reshape
        # x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        # # x_p1 = self.sigmoid(x_p1)
        # x_p1 = self.softmax(x_p1)
        # x_p1 = x * x_p1 + x
        
        # nt, c, h, w = x_shift.size()
        # x_p1 = x_shift.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous()
        # x_p1 = x_p1.mean(1, keepdim=True)
        # # x_p1 = self.action_p1_conv1(x_p1)
        # x_p1_1 = self.action_p11_conv1(x_p1)
        # x_p1_2 = self.action_p12_conv1(x_p1)
        # x_p1_3 = self.action_p13_conv1(x_p1)
        # # x_p1_4 = self.action_p14_conv1(x_p1)
        # x_p1 = torch.cat((x_p1_1, x_p1_2, x_p1_3), dim=1)
        # x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        # x_p1 = self.sigmoid(x_p1)
        # x_p1 = x_shift * x_p1 + x_shift


        # 2D convolution: c*T*1*1, channel excitation
        # x_p2 = self.avg_pool(x_shift)
        # x_p2 = self.avg_pool(x)
        # x_p2 = self.action_p2_squeeze(x_p2)
        # nt, c, h, w = x_p2.size()
        # x_p2 = x_p2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        # x_p2 = self.action_p2_conv1(x_p2)
        # x_p2 = self.relu(x_p2)
        # x_p2 = x_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        # x_p2 = self.action_p2_expand(x_p2)
        # x_p2 = self.sigmoid(x_p2)
        # # x_p2 = self.softmax(x_p2)
        # x_p2 = x * x_p2 + x
        # x_p2 = x_shift * x_p2 + x_shift
               
        
        #=====================类CBAM======================
        x_p2_1 = self.avg_pool(x)
        x_p2_1 = self.action_p2_squeeze(x_p2_1)
        nt, c, h, w = x_p2_1.size()
        x_p2_1 = x_p2_1.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()  #   n, c, t
        x_p2_1 = self.action_p2_conv1(x_p2_1)
        x_p2_1 = self.relu(x_p2_1)
        x_p2_1 = x_p2_1.transpose(2,1).contiguous().view(-1, c, 1, 1)
        x_p2_1 = self.action_p2_expand(x_p2_1)
        # _, c, h, w = x_p2_1.size()
        # x_p2_1 = x_p2_1.view(n_batch, self.n_segment, c, h, w)
        x_p2_1 = self.softmax(x_p2_1)
        # x_p2_1 = x_p2_1.view(-1, c, h, w)
        x_p2_1 = x_p2_1 * x
        # x_p2_1 = x_p2_1 * x + x # NT, C, H, W

        x_p2_2 = self.max_pool(x)
        x_p2_2 = self.action_p2_squeeze(x_p2_2)
        nt, c, h, w = x_p2_2.size()
        x_p2_2 = x_p2_2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        x_p2_2 = self.action_p2_conv1(x_p2_2)
        x_p2_2 = self.relu(x_p2_2)
        x_p2_2 = x_p2_2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        x_p2_2 = self.action_p2_expand(x_p2_2)
        # _, c, h, w = x_p2_2.size()
        # x_p2_2 = x_p2_2.view(n_batch, self.n_segment, c, h, w)
        x_p2_2 = self.softmax(x_p2_2)
        # x_p2_2 = x_p2_2.view(-1, c, h, w)
        x_p2_2 = x_p2_2 * x
        # x_p2_2 = x_p2_2 * x + x
        
        # =====CBAM use=====特征相加再激活
        # x_p2 = x_p2_1 + x_p2_2
        # x_p2 = self.sigmoid(x_p2)
        # x_p2 = x * x_p2
        x_p2 = x_p2_1 + x_p2_2 + x
        # x_p2 = x_p2_1 + x_p2_2   # nt, c, h, w

        #时空注意力
        # reshape
        nt, c, h, w = x_p2.size()
        # x_p1 = x_p2.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous() # n_batch, c, T, h, w
        x_p1_avg = x_p2.mean(1, keepdim=True)   # nt, 1, h, w
        x_p1_max, _ = x_p2.max(1, keepdim=True) # nt, 1, h, w
        x_p1_avg = x_p1_avg.view(n_batch, self.n_segment, 1, h, w).transpose(2,1).contiguous()
        x_p1_max = x_p1_max.view(n_batch, self.n_segment, 1, h, w).transpose(2,1).contiguous()
        x_p1 = torch.cat([x_p1_avg, x_p1_max], dim=1)   # n_batch, 2, 2, h, w
        x_p1 = torch.cat([x_p1_avg, x_p1_max], dim=1)   # n_batch, 2, 2, h, w
        x_p1 = self.action_p1_conv1(x_p1)   # n_batch, 1, t, h, w
        # reshape
        x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)   # nt, 1, h, w
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x_p2 * x_p1 + x_p2
        
        # x_cat = torch.cat((x_p1, x_p2), dim=1)
        # x_cat = self.cat_conv2d(x_cat)
        # # x_cat = self.relu(x_cat)
        # x_cat = self.leackrelu(x_cat)
        # x_cat = x * x_cat + x
        # out = self.net(x_cat)
        # ===============
        # out = self.net(x_p1)
        out = self.net(x_p1 + x)
        # ===============attention

        # out = self.net(x_p1 + x_p2 + x_p3)
        # out = self.net(0.3 * x_p1 + 0.7 * x_p2)
        # out = self.net(x_p1 + 2 * x_p2)
        
        # out = self.net(x_p2)
        # out = self.net(x_shift)

        # ==================== 相邻帧之间做差，然后经过simoid，获取运动位置特征图，和原始相乘==========
        # x_p3 = self.action_p3_squeeze(x)
        # x_p3 = self.action_p3_bn1(x_p3)
        # nt, c, h, w = x_p3.size()
        # x3_plus0, _ = x_p3.view(n_batch, self.n_segment, c, h, w).split([self.n_segment-1, 1], dim=1) # n, t-1, c, h, w
        # _ , x3_plus1 = x_p3.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment-1], dim=1)  # n, t-1, c, h, w
        # x_p3 = x3_plus1 - x3_plus0  # n, t-1, c, h, w
        # # # 第1帧运动信息填充为0, 第0帧到t-1帧的运动信息
        # x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)  # n, t, c, h, w  constant默认填充0
        # x_p3 = x_p3.view(nt, c, h, w)
        # x_p3 = self.action_p3_expand(x_p3)
        # x_p3 = self.sigmoid(x_p3)
        # # x_p3 = x_p3 * x_p1
        # # out = self.net(x_p3 + x_p1)
        # x_p3 = x_p3 * x_p1 + x_p3 * x
        # x_p3 = self.leackrelu(x_p3)
        # out = self.net(x_p3 + x_p1)
        # out = self.net(x_p3 + x_p1 + x)
        # =======================end=========================


        return out





class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))


    # pdb.set_trace()
    # if isinstance(net, SPP):
    #     if place == 'block':
    #         def make_block_temporal(stage, this_segment):
    #             blocks = list(stage.children())
    #             print('=> Processing stage with {} blocks'.format(len(blocks)))
    #             for i, b in enumerate(blocks):
    #                 blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
    #             return nn.Sequential(*(blocks))

    #         pdb.set_trace()
    #         net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
    #         net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
    #         net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
    #         net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

    #     elif 'blockres' in place:
    #         n_round = 1
    #         # 没有用到
    #         if len(list(net.layer3.children())) >= 23:
    #             n_round = 2
    #             print('=> Using n_round {} to insert temporal shift'.format(n_round))
            
    #         # 为每一个blockres添加action模块
    #         def make_block_temporal(stage, this_segment):
    #             blocks = list(stage.children())
    #             print('=> Processing stage with {} blocks residual'.format(len(blocks)))
    #             for i, b in enumerate(blocks):
    #                 if i % n_round == 0:
    #                     blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
    #                     # pdb.set_trace()
    #             return nn.Sequential(*blocks)

    #         # pdb.set_trace()
    #         # 给res2-5添加action模块
    #         net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
    #         net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
    #         net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
    #         net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

    # else:
    #     raise NotImplementedError(place)
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
                return nn.Sequential(*(blocks))

            pdb.set_trace()
            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            # 没有用到
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))
            
            # 为每一个blockres添加action模块
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
                        # pdb.set_trace()
                return nn.Sequential(*blocks)

            # pdb.set_trace()
            # 给res2-5添加action模块
            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError






