from torch import nn

from models.basic_ops import ConsensusModule
from models.spatial_transforms import *
from torch.nn.init import normal_, constant_
import torchvision
import torch
import pdb


class TSN(nn.Module):
    # 初始化模型，设置参数
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, img_feature_dim=224,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False, module=None):
        super(TSN, self).__init__()
        # 输入模态，RGB或光流
        self.modality = modality
        # 一个video分多少段，对应论文中的K
        self.num_segments = num_segments
        self.reshape = True
        # 是否在softmax前融合
        self.before_softmax = before_softmax
        self.dropout = dropout
        # 数据集修改的类别，没有用到
        self.crop_num = crop_num
        # 选择聚合函数，默认为平均池化（avg）
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain
        # 判断是否加入shift
        self.is_shift = is_shift
        # shift操作的比例函数
        self.shift_div = shift_div
        # 判断标志，当为blockres表示嵌入残差模块，当为block表示嵌入直链模块
        self.shift_place = shift_place
        self.base_model_name = base_model
        # 学习率优化策略中队第五个全连接层进行参数调整的判断标志，默认false
        self.fc_lr5 = fc_lr5
        # 表示是否在时间维度进行池化降维，相应的2，3，4层num segment减半，默认false
        self.temporal_pool = temporal_pool
        # 判断是都加入non_local网络模块，默认false
        self.non_local = non_local

        self.module = module

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")
        # new_length 视频提帧起点，RGB默认为1
        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        # 是否打印bn
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    
    def _prepare_tsn(self, num_class):
        '''
            对经过_prepare_base_model函数处理过后的模型(basemodel)进行进一步修改
            微调全连接层的结构。
        '''
        # getattr是获得属性值，一般可以用来获取网络结构相关的信息，
        # 输入包含2个值，分别是基础网络和要获取值的属性名。
        # 获取网络最后一层的输入feature_map的通道数，存入featrue_dim
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        # feature_dim = 256

        # 判断是否有dropout，如果有，则添加一个dropout后再添加一个全连接层，否则直接连接全连接层。
        if self.dropout == 0:
            # setattr是torch.nn.Module类的一个方法，用来为输入的某个属性赋值，
            # 一般可以用来修改网络结构，输入包含3个值，分别是基础网络，要赋值的属性名，要赋的值。
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            # setattr(object,name,value)：设置object 对象的 name 属性设为 value
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            # 全连接
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        # 对全连接层的网络权重进行0均值且指定标准差初始化操作，之后对偏差初始化为0
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            # hasattr(object,name)：检查 object 对象是否包含名为 name 的属性或方法，
            # 有返回True，否则返回False。
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        '''
            是否进行shift操作和添加non_local网络模块
            选择主体网络结构并进行数据预处理设置
            对输入数据集的某些参数进行调整。
        '''
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:

            # from models.spp import make_spp
            # self.base_model = make_spp(base_model)
            
            # 根据base_model的不同指定值来导入不同的网络，对不同基础模型设定不同的输入尺寸、均值和方差
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                print('Adding action...')
                from models.action import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            elif self.module == "CBAM":
                print('Adding CBAM...')
                from models.resnet_CBAM import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "SE":
                print('Adding CBAM...')
                from models.SE_avg import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KeyFrame":
                print('Adding CBAM...')
                from models.Keyframe import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            elif self.module == "KeyFrame2":
                print('Adding CBAM...')
                from models.Keyframe2 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "SC":
                print('Adding sc...')
                from models.SC import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "CS":
                print('Adding CS...')
                from models.CS import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF_CS":
                print('Adding KF_CS...')
                from models.KF_CS_3 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF_SC":
                print('Adding KF_SC...')
                from models.KF_SC import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "CS_KF":
                print('Adding CS_KF...')
                from models.TSC_KF import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            
            elif self.module == "KF_CS_2":
                print('Adding KF_CS...')
                from models.KF_CS_2 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF2_4_1":
                print('Adding KF method 4_1...')
                from models.KF2_4_1 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF2_CS_1":
                print('Adding KF_CS 1...')
                from models.KF2_CS_1 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF2_CS_1_1":
                print('Adding KF2_CS 1_1...')
                from models.KF2_CS_1_1 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            
            elif self.module == "KF2_CS_2":
                print('Adding KF_CS...')
                from models.KF2_CS_2 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF2_CS_2_0":
                print('Adding KF2_CS_2_0...')
                from models.KF2_CS_2_0 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            
            elif self.module == "KF2_CS_2_1":
                print('Adding KF_CS...')
                from models.KF2_CS_2_1 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            

            elif self.module == "KF2_CS_3":
                print('Adding KF_CS...')
                from models.KF2_CS_3 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF2_CS_4":
                print('Adding KF2_CS_4...')
                from models.KF2_CS_4 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF2_CS_4_1":
                print('Adding KF2_CS_4_1...')
                from models.KF2_CS_4_1 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF3_CS_3":
                print('Adding KF_CS...')
                from models.KF3_CS_3 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF3_CS_3_1":
                print('Adding KF_CS...')
                from models.KF3_CS_3_1 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            
            elif self.module == "KF3_CS_5":
                print('Adding KF_CS...')
                from models.KF3_CS_5 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF3_CS_5_1":
                print('Adding KF_CS...')
                from models.KF3_CS_5_1 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "KF3_CS_5_2":
                print('Adding KF_CS...')
                from models.KF3_CS_5_2 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "res2":
                print('Adding KF2_CS4_1...')
                from models.res2 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)    

            elif self.module == "res3":
                print('Adding KF2_CS4_1...')
                from models.res3 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)   

            elif self.module == "res4":
                print('Adding KF2_CS4_1...')
                from models.res4 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)    
            elif self.module == "res4_new":
                print('Adding KF2_CS4_1...')
                from models.res4_new import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)    

            elif self.module == "res5":
                print('Adding KF2_CS4_1...')
                from models.res5 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)      

            elif self.module == "res2_3":
                print('Adding KF2_CS4_1...')
                from models.res2_3 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)       

            elif self.module == "res2_3":
                print('Adding KF2_CS4_1...')
                from models.res2_3 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "res2_4":
                print('Adding KF2_CS4_1...')
                from models.res2_4 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "res_2_4":
                print('Adding KF2_CS4_1...')
                from models.res_2_4 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "res3_4":
                print('Adding KF2_CS4_1...')
                from models.res3_4 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "res4_5":
                print('Adding KF2_CS4_1...')
                from models.res4_5 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "res3_4_5":
                print('Adding KF2_CS4_1...')
                from models.res3_4_5 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            elif self.module == "res2_3_4":
                print('Adding KF2_CS4_1...')
                from models.res2_3_4 import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)  

            elif self.module == "res2_3_4_5_new":
                from models.res2_3_4_5_new import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)    

            elif self.module == "TC_TS_sum":
                from models.TC_TS_sum import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool) 
            
            elif self.module == "TC_TS_cat":
                from models.TC_TS_cat import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool) 
            
            elif self.module == "TCS":
                from models.TCS import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool) 

            elif self.module == "KF_CS_sum":
                from models.KF_CS_sum import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool) 

            elif self.module == "KF_CS_cat":
                from models.KF_CS_cat import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            elif self.module == "TC_TS_feature_cat":
                from models.TC_TS_feature_cat import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)


            # 没有用到
            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            # self.input_mean = [0.485, 0.456, 0.406]
            # self.input_std = [0.229, 0.224, 0.225]
            # 最后一层
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif 'res2net' in base_model:
            from archs.res2net import res2net50_26w_4s
            self.base_model = res2net50_26w_4s(True if self.pretrain == 'imagenet' else False)
            if self.is_shift:
                from models.temporal_shift_res2net import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)
            
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length


        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)
            # self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:   
                from models.action import Action
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        m.conv[0] = Action(m.conv[0], n_segment=self.num_segments, shift_div=self.shift_div)           

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            if self.is_shift:
                from archs.bn_inception_action import bninception
                self.base_model = bninception(pretrained=self.pretrain, n_segment=self.num_segments, fold_div=self.shift_div)
                self.input_size = self.base_model.input_size
                self.input_mean = self.base_model.mean
                self.input_std = self.base_model.std
                self.base_model.last_layer_name = 'fc'
                if self.modality == 'Flow':
                    self.input_mean = [128]
                elif self.modality == 'RGBDiff':
                    self.input_mean = self.input_mean * (1 + self.new_length)
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    # 重写train函数，冻结除第一次的其他BN层
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    # 获取模型的每一层并保存参数用于优化
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_weight = []
        custom_bn = []

        conv_cnt = 0
        bn_cnt = 0
        for name, m in self.named_modules():
            if 'action' in name:
                ps = list(m.parameters())
                if 'bn' not in name:
                    custom_weight.append(ps[0])
                    if len(ps) == 2:
                        pdb.set_trace()
                else:
                    if not self._enable_pbn or bn_cnt == 1:
                        custom_bn.extend(list(m.parameters()))

            else:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm3d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_weight"},
            {'params': custom_bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "custom_bn"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    # 向前传播
    def forward(self, input, no_reshape=False):
        # pdb.set_trace()
        assert input.size()[1] > 3, 'channel and temporal dimension mismatch, tensor size should be: n_batch, n_segment, nc, h, w'

        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)
            
            # input: nb, n_segment, c, h, w
            # baseout size [60, 2048]
            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:])) # nb*n_segment, c, h, w
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)    # size [60, 3]

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # reshape默认True
        if self.reshape: 
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                # pdb.set_trace()
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:]) 
            # size [2, 30, 3]
            output = self.consensus(base_out)   # 1,1,3
            return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
    
    # 根据输入不同，获取不同的数据预处理操作
    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])




