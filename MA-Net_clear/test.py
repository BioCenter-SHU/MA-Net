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
from zmq import device


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
    parser.add_argument('--cuda_id', type=str, default='0')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, 
                    default='/YOUR_PATH/Clssification/DuoChiDu/checkpoint/resnet50/CS/resnet50_30/2022-12-07-11-24-15/resnet50_30/clip_len_30_0_checkpoint.pth.tar', 
                    help='the path of checkpoint')
    parser.add_argument('--frame_path', type=str, default="/YOUR_PATH/MyDataset/0_single_plaque/video_302_frame_120/", 
                    help='the path of frame')
    parser.add_argument('--root_path', type=str, default='/YOUR_PATH/Clssification/DuoChiDu/annotation/avg_K5/0_val.txt', 
                    help='the path of testing txt')

    # args for dataloader
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--clip_len', type=int, default=30)
    
    # args for preprocessing
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', action="store_true")
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--dataset', default='sthv2', type=str)

    # args for testing 
    parser.add_argument('--test_crops', default=1, type=int)
    parser.add_argument('--scale_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--clip_num', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--module', type=str, default='CS')

    args = parser.parse_args()
    return args

args = parse_opts()

params = dict()
params['num_classes'] = 3


# os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
# device = 'cuda:0'
device = 'cpu'


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



def inference(k, model, val_dataloader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # 评价指标
    top1 = AverageMeter()
    top5 = AverageMeter()
    # micro
    micro_acc = torchmetrics.Accuracy()
    micro_recall = torchmetrics.Recall()
    micro_precision = torchmetrics.Precision()
    micro_auc = torchmetrics.AUROC(average="macro", num_classes=3)
    micro_f1 = torchmetrics.F1Score()
    micro_acc = micro_acc.to(device, non_blocking=True).float()
    micro_recall = micro_recall.to(device, non_blocking=True).float()
    micro_precision = micro_precision.to(device, non_blocking=True).float()
    micro_auc = micro_auc.to(device, non_blocking=True).float()
    micro_f1 = micro_f1.to(device, non_blocking=True).float()
    # macro
    macro_acc = torchmetrics.Accuracy(average='macro', num_classes=3)
    macro_recall = torchmetrics.Recall(average='macro', num_classes=3)
    macro_precision = torchmetrics.Precision(average='macro', num_classes=3)
    macro_auc = torchmetrics.AUROC(average="macro", num_classes=3)
    macro_f1 = torchmetrics.F1Score(average="macro", num_classes=3)
    macro_acc = macro_acc.to(device, non_blocking=True).float()
    macro_recall = macro_recall.to(device, non_blocking=True).float()
    macro_precision = macro_precision.to(device, non_blocking=True).float()
    macro_auc = macro_auc.to(device, non_blocking=True).float()
    macro_f1 = macro_f1.to(device, non_blocking=True).float()
    # weighted
    weighted_acc = torchmetrics.Accuracy(average='weighted', num_classes=3)
    weighted_recall = torchmetrics.Recall(average='weighted', num_classes=3)
    weighted_precision = torchmetrics.Precision(average='weighted', num_classes=3)
    weighted_auc = torchmetrics.AUROC(average="weighted", num_classes=3)
    weighted_f1 = torchmetrics.F1Score(average='weighted', num_classes=3)
    weighted_acc = weighted_acc.to(device, non_blocking=True).float()
    weighted_recall = weighted_recall.to(device, non_blocking=True).float()
    weighted_precision = weighted_precision.to(device, non_blocking=True).float()
    weighted_auc = weighted_auc.to(device, non_blocking=True).float()
    weighted_f1 = weighted_f1.to(device, non_blocking=True).float()

    model.eval()
    # 记录测试数据的结果
    result_path = 'test_result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    f = open('test_result/{}_result.csv'.format(args.module),'w')
    csv_write = csv.writer(f)
    csv_write.writerow(['video','0','1','2','pre','label','prec'])

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
                print(outputs.data)
                print(outputs.data.mean(1))
                
            labels = labels.to(device, non_blocking=True).long()
            pre_list = outputs.mean(1).argmax(1).cpu().numpy().tolist()
            # micro
            micro_acc(outputs.mean(1).argmax(1), labels)
            micro_auc.update(outputs.mean(1), labels)
            micro_recall(outputs.mean(1).argmax(1), labels)
            micro_precision(outputs.mean(1).argmax(1), labels)
            micro_f1.update(outputs.mean(1), labels)
            # macro
            macro_acc(outputs.mean(1).argmax(1), labels)
            macro_auc.update(outputs.mean(1), labels)
            macro_recall(outputs.mean(1).argmax(1), labels)
            macro_precision(outputs.mean(1).argmax(1), labels)
            macro_f1.update(outputs.mean(1), labels)
            # weighted
            weighted_acc(outputs.mean(1).argmax(1), labels)
            weighted_auc.update(outputs.mean(1), labels)
            weighted_recall(outputs.mean(1).argmax(1), labels)
            weighted_precision(outputs.mean(1).argmax(1), labels)
            weighted_f1.update(outputs.mean(1), labels)

            # pre_list = outputs.max(1).argmax(1).cpu().numpy().tolist()
            # # micro
            # micro_acc(outputs.max(1).argmax(1), labels)
            # micro_auc.update(outputs.max(1), labels)
            # micro_recall(outputs.max(1).argmax(1), labels)
            # micro_precision(outputs.max(1).argmax(1), labels)
            # micro_f1.update(outputs.max(1), labels)
            # # macro
            # macro_acc(outputs.max(1).argmax(1), labels)
            # macro_auc.update(outputs.max(1), labels)
            # macro_recall(outputs.max(1).argmax(1), labels)
            # macro_precision(outputs.max(1).argmax(1), labels)
            # macro_f1.update(outputs.mean(1), labels)
            # # weighted
            # weighted_acc(outputs.max(1).argmax(1), labels)
            # weighted_auc.update(outputs.max(1), labels)
            # weighted_recall(outputs.max(1).argmax(1), labels)
            # weighted_precision(outputs.max(1).argmax(1), labels)
            # weighted_f1.update(outputs.max(1), labels)

            # test_loss /= num_batches
            # measure accuracy and record loss
            # prec1, prec5 = accuracy(outputs.data.mean(1), labels, topk=(1, 3))
            # # prec1, prec5 = accuracy(outputs.data.max(1), labels, topk=(1, 3))
            # top1.update(prec1.item(), labels.size(0))
            # top5.update(prec5.item(), labels.size(0))
            batch_time.update(time.time() - end)
            outputs_list = outputs.data.mean(1).cpu().numpy().tolist()
            # outputs_list = outputs.data.max(1).cpu().numpy().tolist()
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
    
    # micro
    micro_acc = micro_acc.compute()
    micro_recall = micro_recall.compute()
    micro_precision = micro_precision.compute()
    micro_auc = micro_auc.compute()
    micro_f1 = micro_f1.compute()
    # macro
    macro_acc = macro_acc.compute()
    macro_recall = macro_recall.compute()
    macro_precision = macro_precision.compute()
    macro_auc = macro_auc.compute()
    macro_f1 = macro_f1.compute()
    # weighted
    weighted_acc = weighted_acc.compute()
    weighted_recall = weighted_recall.compute()
    weighted_precision = weighted_precision.compute()
    weighted_auc = weighted_auc.compute()
    weighted_f1 = weighted_f1.compute()

    print_string = ('Top-1: {top1_acc:.2f}, ' 'Top-5: {top5_acc:.2f}'.format(
        top1_acc=top1.avg,
        top5_acc=top5.avg)
        )
    print(print_string)

    print("[micro] "
                f"acc: {(100 * micro_acc):>0.2f}%, "
                f"recall: {(100 * micro_recall):>0.2f}%, "
                f"precision: {(100 * micro_precision):>8f}, "
                f"auc: {(100 * micro_auc.item()):>0.2f}%, "
                f"f1: {(100 * micro_f1.item()):>0.2f}%")

    print("[macro] "
                f"acc: {(100 * macro_acc):>0.2f}%, "
                f"recall: {(100 * macro_recall):>0.2f}%, "
                f"precision: {(100 * macro_precision):>8f}, "
                f"auc: {(100 * macro_auc.item()):>0.2f}%, "
                f"f1: {(100 * macro_f1.item()):>0.2f}%")
    
    print("[weighted] "
                f"acc: {(100 * weighted_acc):>0.2f}%, "
                f"recall: {(100 * weighted_recall):>0.2f}%, "
                f"precision: {(100 * weighted_precision):>8f}, "
                f"auc: {(100 * weighted_auc.item()):>0.2f}%, "
                f"f1: {(100 * weighted_f1.item()):>0.2f}%, ")

    f.close()


if __name__ == '__main__':
    # if args.test_crops == 1:
    #     cropping = torchvision.transforms.Compose([
    #         GroupScale(args.scale_size),
    #         # GroupCenterCrop(args.crop_size)
    #     ])
    # elif args.test_crops == 3:
    #     cropping = torchvision.transforms.Compose([
    #         GroupFullResSample(args.crop_size, args.scale_size, flip=False)
    #     ])
    # elif args.test_crops == 5: 
    #     cropping = torchvision.transforms.Compose([
    #         GroupOverSample(args.crop_size, args.scale_size, flip=False)
    #     ])
   


    checkpoint_path = args.checkpoint
    cudnn.benchmark = True
    model = TSN_model.TSN(params['num_classes'], args.clip_len, 'RGB', 
                        is_shift = args.is_shift,
                        base_model=args.base_model, 
                        shift_div = args.shift_div, 
                        img_feature_dim = args.crop_size,
                        consensus_type='avg',
                        fc_lr5 = True,
                        module=args.module)

    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    print("load checkpoint {}".format(checkpoint_path))
    model.load_state_dict(pretrained_dict['state_dict'])
    # model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)

    val_dataloader = DataLoader(dataset.dataset_video_inference(root_path=args.root_path, 
                                                                    frame_path=args.frame_path, 
                                                                    clip_num=args.clip_num, clip_len=args.clip_len),
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers)
    inference(0, model, val_dataloader)


    # for i in range(args.k):
    #     val_dataloader = DataLoader(dataset.dataset_video_inference(root_path='annotation/test_{}.txt'.format(i), 
    #                                                                 frame_path='/YOUR_PATH/dataset/train+test/', 
    #                                                                 clip_num=args.clip_num, clip_len=args.clip_len),
    #                                     batch_size=args.batch_size, 
    #                                     num_workers=args.num_workers)
    
    #     inference(i, model, val_dataloader)   

