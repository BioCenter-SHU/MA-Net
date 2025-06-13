import argparse
from yaml import parse

from zmq import proxy_steerable
import models

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=str, default='1')


parser.add_argument('--module', type=str, default='')

# args for dataloader
parser.add_argument('--is_train', action="store_true")
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--clip_len', type=int, default=30)
parser.add_argument('--k', type=int, default=5, help='k-fold')
parser.add_argument('--frame_path', type=str, default='/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/', 
                    help='the path of frame')
parser.add_argument('--image_dir', type=str, default='data/KFold_data.txt',
                    help='the path of video txt')
parser.add_argument('--save_path', type=str, default='SPP',
                    help='cheackpoint logs save path')


# args for preprocessing
parser.add_argument('--initial_scale', type=float, default=1,
                    help='Initial scale for multiscale cropping')
parser.add_argument('--n_scales', default=5, type=int,
                    help='Number of scales for multiscale cropping')
parser.add_argument('--scale_step', default=0.84089641525, type=float,
                    help='Scale step for multiscale cropping')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', type=float, default=[30,60,80], nargs="+",
                    help='lr steps for decreasing learning rate') 
parser.add_argument('--clip_gradient', '--gd', type=int, default=20, help='gradient clip')
parser.add_argument('--shift_div', default=8, type=int)
parser.add_argument('--is_shift', action="store_true", help='use action module')
parser.add_argument('--npb', action="store_true")
parser.add_argument('--pretrain', type=str, default='imagenet') # 'imagenet' or False
parser.add_argument('--dropout', default=0.9, type=float)
parser.add_argument('--base_model', default='resnet50', type=str)
parser.add_argument('--dataset', default='EgoGesture', type=str)
parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                help='number of total epochs to run')
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--recover_from_checkpoint', default=None, type=str)
parser.add_argument('--display', type=int, default=20)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--frame_sample_rate', type=int, default=1)