import os
import csv
import pdb
import time
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision
from tqdm import tqdm, trange
import torchmetrics


from models.spatial_transforms import *
from models.temporal_transforms import *
from data import dataset_jester, dataset_EgoGesture, dataset_sthv2, dataset
import utils as utils
from models import models as TSN_model
import argparse

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore")



def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='2')

    # args for dataloader
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--clip_len', type=int, default=8)
    
    # args for preprocessing
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', action="store_true")
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--dataset', default='EgoGesture', type=str)

    # args for testing 
    parser.add_argument('--test_crops', default=1, type=int)
    parser.add_argument('--scale_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--clip_num', type=int, default=10)

    args = parser.parse_args()
    return args

args = parse_opts()

params = dict()
if args.dataset == 'EgoGesture':
    params['num_classes'] = 83
elif args.dataset == 'jester':
    params['num_classes'] = 27
elif args.dataset == 'sthv2':
    params['num_classes'] = 3



annot_path = 'data/{}_annotation'.format(args.dataset)
label_path = '/home/raid/zhengwei/{}/'.format(args.dataset) # for submitting testing results
# annot_path = '/home/raid/zhengwei/kinetic-700'

os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
device = 'cuda:0'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def inference(model, val_dataloader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    test_acc = torchmetrics.Accuracy(average='weighted', num_classes=3)
    test_recall = torchmetrics.Recall(average='weighted', num_classes=3)
    test_precision = torchmetrics.Precision(average='weighted', num_classes=3)
    test_auc = torchmetrics.AUROC(average="weighted", num_classes=3)
    test_f1 = torchmetrics.F1Score(average='weighted', num_classes=3)
    test_acc = test_acc.to(device, non_blocking=True).float()
    test_recall = test_recall.to(device, non_blocking=True).float()
    test_precision = test_precision.to(device, non_blocking=True).float()
    test_auc = test_auc.to(device, non_blocking=True).float()
    test_f1 = test_f1.to(device, non_blocking=True).float()
    size = len(val_dataloader.dataset)
    num_batches = len(val_dataloader)
    # test_loss, correct = 0, 0
    model.eval()

    f = open('result/test_result.csv','w')
    csv_write = csv.writer(f)
    csv_write.writerow(['video','1','2','3','pre','label','prec'])

    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(val_dataloader)):
            data_time.update(time.time() - end)
            if args.dataset == 'EgoGesture':
                rgb, depth, labels = inputs[0], inputs[1], inputs[2]
                rgb = rgb.to(device, non_blocking=True).float()
                depth = depth.to(device, non_blocking=True).float()
                nb, n_clip, nt, nc, h, w = rgb.size()
                rgb = rgb.view(-1, nt//args.test_crops, nc, h, w) # n_clip * nb (1) * crops, T, C, H, W
                outputs = model(rgb)
                outputs = outputs.view(nb, n_clip*args.test_crops, -1)
                outputs = F.softmax(outputs, 2)
            else:
                # pdb.set_trace()
                rgb, labels = inputs[0], inputs[1]
                rgb = rgb.to(device, non_blocking=True).float()
                nb, n_clip, nt, nc, h, w = rgb.size()
                rgb = rgb.view(-1, nt//args.test_crops, nc, h, w)
                outputs = model(rgb)
                outputs = outputs.view(nb, n_clip*args.test_crops, -1)
                outputs = F.softmax(outputs, 2)
                print(outputs.data.mean(1))
                
            labels = labels.to(device, non_blocking=True).long()
            pre_list = outputs.mean(1).argmax(1).cpu().numpy().tolist()
            test_acc(outputs.mean(1).argmax(1), labels)
            test_auc.update(outputs.mean(1), labels)
            test_recall(outputs.mean(1).argmax(1), labels)
            test_precision(outputs.mean(1).argmax(1), labels)
            test_f1.update(outputs.mean(1), labels)
            # test_loss /= num_batches
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data.mean(1), labels, topk=(1, 3))
            top1.update(prec1.item(), labels.size(0))
            top5.update(prec5.item(), labels.size(0))
            batch_time.update(time.time() - end)
            outputs_list = outputs.data.mean(1).cpu().numpy().tolist()
            label_list = labels.cpu().numpy().tolist()
            csv_write.writerow([step, outputs_list[0][0],outputs_list[0][1],outputs_list[0][2],label_list[0], pre_list[0]])
            end = time.time()

            if (step+1) % 100 == 0:
                print_string = ('Top-1: {top1_acc.avg:.2f}, ' 
                                'Top-5: {top5_acc.avg:.2f}'
                                .format(
                                        top1_acc=top1,
                                        top5_acc=top5)
                                )
                print(print_string) 
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    total_f1 = test_f1.compute()
    print_string = ('Top-1: {top1_acc:.2f}, ' 'Top-5: {top5_acc:.2f}'.format(
        top1_acc=top1.avg,
        top5_acc=top5.avg)
        )
    print(print_string)

    print(f"acc: {(100 * total_acc):>0.2f}%, "
                f"recall: {(100 * total_recall):>0.2f}%, "
                f"precision: {(100 * total_precision):>8f}, "
                f"auc: {(100 * total_auc.item()):>0.2f}%, "
                f"f1: {(100 * total_f1.item()):>0.2f}%, ")

    f.close()


if __name__ == '__main__':
    if args.dataset == 'EgoGesture':
        cropping = torchvision.transforms.Compose([
            GroupScale([224, 224])
        ])
    else:
        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose([
                GroupScale(args.scale_size),
                GroupCenterCrop(args.crop_size)
            ])
        elif args.test_crops == 3:
            cropping = torchvision.transforms.Compose([
                GroupFullResSample(args.crop_size, args.scale_size, flip=False)
            ])
        elif args.test_crops == 5: 
            cropping = torchvision.transforms.Compose([
                GroupOverSample(args.crop_size, args.scale_size, flip=False)
            ])


    # input_mean=[.485, .456, .406]
    # input_std=[.229, .224, .225]
    # normalize = GroupNormalize(input_mean, input_std)


    # # for mulitple clip test, use random sampling;
    # # for single clip test, use middle sampling  
    # spatial_transform  = torchvision.transforms.Compose([
    #                         cropping,
    #                         Stack(),
    #                         ToTorchFormatTensor(),
    #                         normalize
    #                         ])
    # temporal_transform = torchvision.transforms.Compose([
    #         TemporalUniformCrop_train(args.clip_len)
    #     ])    


    checkpoint_path = 'checkpoint/2022-05-27-16-44-10/clip_len_8_3_Fold_checkpoint.pth.tar'
    cudnn.benchmark = True
    model = TSN_model.TSN(params['num_classes'], args.clip_len, 'RGB', 
                        is_shift = args.is_shift,
                        base_model=args.base_model, 
                        shift_div = args.shift_div, 
                        img_feature_dim = args.crop_size,
                        consensus_type='avg',
                        fc_lr5 = True)

    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    print("load checkpoint {}".format(checkpoint_path))
    model.load_state_dict(pretrained_dict['state_dict'])
    # model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)


    # if args.dataset == 'jester':
    #     val_dataloader = DataLoader(dataset_jester.dataset_video_inference(annot_path, 'val', clip_num=args.clip_num, 
    #                                                                         spatial_transform=spatial_transform, 
    #                                                                         temporal_transform = temporal_transform),
    #                                 batch_size=args.batch_size, 
    #                                 num_workers=args.num_workers)
    # elif args.dataset == 'sthv2':
    #     # val_dataloader = DataLoader(dataset_sthv2.dataset_video_inference(annot_path, 'val', clip_num=args.clip_num, 
    #     #                                                                         spatial_transform=spatial_transform, 
    #     #                                                                         temporal_transform = temporal_transform),
    #     #                             batch_size=args.batch_size, 
    #     #                             num_workers=args.num_workers)
        

    # elif args.dataset == 'EgoGesture':
    #     val_dataloader = DataLoader(dataset_EgoGesture.dataset_video_inference(annot_path, 'test', clip_num=args.clip_num, 
    #                                                                             spatial_transform=spatial_transform, 
    #                                                                             temporal_transform = temporal_transform),
    #                                 batch_size=args.batch_size, 
    #                                 num_workers=args.num_workers)

    val_dataloader = DataLoader(dataset.dataset_video_inference(root_path='annotation/test_1.txt', 
                                                                frame_path='/YOUR_PATH/dataset/train+test/', 
                                                                clip_num=args.clip_num, clip_len=args.clip_len),
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers)
    
    inference(model, val_dataloader)   

