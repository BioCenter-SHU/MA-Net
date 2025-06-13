from turtle import forward
from numpy import outer, pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from models.spp import SPP

class Action(nn.Module):
    '''
        单跑一个keyframe模块
    '''
    def __init__(self, net, n_segment=3, shift_div=8):
        super(Action, self).__init__()
        self.net = net
        self.n_segment = n_segment
        # keyframe attention
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.keyframe_fc1 = nn.Linear(self.n_segment, self.n_segment // 2, bias=False)
        self.kf_relu = nn.ReLU(inplace=True)
        self.keyframe_fc2 = nn.Linear(self.n_segment // 2, self.n_segment, bias=False)
        self.keyframe_softmax = nn.Softmax(dim=1)

        print('====USE keyframe method 4_1 Block====')
        

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        
        # ==================keyframe====================
        x_kf = x.view(n_batch, self.n_segment, c, h, w)
        x_kf = self.global_max_pool(x).view(n_batch, -1)   # n, t
        x_kf = self.keyframe_fc1(x_kf)  # n, t//2
        x_kf = self.kf_relu(x_kf)
        x_kf = self.keyframe_fc2(x_kf)  # n, t
        x_kf = self.keyframe_softmax(x_kf).view(nt, 1, 1, 1)  # nt, 1, 1, 1
        x_kf = x_kf * x 

        out = self.net(x_kf + x)
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