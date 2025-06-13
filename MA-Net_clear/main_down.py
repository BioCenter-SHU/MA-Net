from ast import arg
import os
import csv
import pdb
import time
from unicodedata import name
import numpy as np
from sklearn.metrics import auc
from sklearn.preprocessing import scale
import torch
from torch import nn, optim, tensor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision
from tqdm import tqdm, trange
# from zmq import device
import random


from models.spatial_transforms import *
from models.temporal_transforms import *
from data import dataset
import utils as utils
from models import models as TSN_model
from opts import parser

from PIL import Image
import torchmetrics

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


import warnings
warnings.filterwarnings("ignore")

args = parser.parse_args()

# 优化训练速度
# 和mian.py相比，修改：
# （1）删掉了train过程中的auc，f1等指标的计算，只保留top1和loss
# （2）修改随机种子位置, cudnn.benchmark = False
# （3）修改为混合精度


#------------------CUDA for pytorch---------------------
# 单张显卡
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 多张显卡
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
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def seed_torch(seed=42):
    # #----------------可复现，随机种子------------------------
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # 为CPU中设置种子，生成随机数
    torch.manual_seed(seed)
    # 为特定GPU设置种子，生成随机数：
    # torch.cuda.manual_seed(seed)
    # 为所有GPU设置种子，生成随机数：  
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.benchmark = False
    cudnn.deterministic = True
    

def train_epoch(k, model, train_dataloader, epoch, criterion, optimizer, writer, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_batches = len(train_dataloader)

    model.train()
    end = time.time()
    for step, inputs in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        rgb, labels = inputs[0], inputs[1]  # rgb: [n, t, c, h, w], labels size=1
        rgb = rgb.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        # outputs == pre, X== rgb, y == labels
        with autocast():
            outputs = model(rgb)    # 2, 3 
            loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 3))
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1.item(), labels.size(0))
        top5.update(prec5.item(), labels.size(0))

        # optimizer.zero_grad()
        # loss.backward()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        # optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if (step+1) % 10 == 0:
            print_string = ('Fold: [{0}], Epoch: [{1}][{2}/{3}], lr: {lr:.8f}, '
                             'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                             'loss: {loss.val:.4f} ({loss.avg:.4f}), '
                             'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                             'Top-5: {top5_acc.val:.2f} ({top5_acc.avg:.2f})'
                             .format(k, epoch, step+1, num_batches,
                                      lr = optimizer.param_groups[2]['lr'],
                                      data_time = data_time, batch_time=batch_time,
                                      loss = losses, top1_acc = top1, top5_acc = top5
                                      )
                            )
            print(print_string)

    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)


def validation(k, model, val_dataloader, epoch, criterion,  writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # 实例化相关metrics的计算对象
    test_acc = torchmetrics.Accuracy()
    test_recall = torchmetrics.Recall(average='macro', num_classes=3)
    test_precision = torchmetrics.Precision(average='macro', num_classes=3)
    test_auc = torchmetrics.AUROC(average="macro", num_classes=3)
    test_f1 = torchmetrics.F1Score(average="macro", num_classes=3)
    test_acc = test_acc.to(device, non_blocking=True).float()
    test_recall = test_recall.to(device, non_blocking=True).float()
    test_precision = test_precision.to(device, non_blocking=True).float()
    test_auc = test_auc.to(device, non_blocking=True).float()
    test_f1 = test_f1.to(device, non_blocking=True).float()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            rgb, labels = inputs[0], inputs[1]
            rgb = rgb.to(device, non_blocking=True).float()
            outputs = model(rgb)
            labels = labels.to(device, non_blocking=True).long()

            loss = criterion(outputs, labels)
       
            # 一个batch进行计算迭代
            test_acc(outputs.argmax(1), labels)
            test_auc.update(outputs, labels)
            test_recall(outputs.argmax(1), labels)
            test_precision(outputs.argmax(1), labels)
            test_f1.update(outputs, labels)
            # test_loss /= num_batches

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 3))
            losses.update(loss.item(), labels.size(0))
            top1.update(prec1.item(), labels.size(0))
            top5.update(prec5.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % 10 == 0:
                print_string = ('Test: [{0}][{1}][{2}], '
                                'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                                'loss: {loss.val:.4f} ({loss.avg:.4f}), '
                                'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                                'Top-5: {top5_acc.val:.2f} ({top5_acc.avg:.2f})'
                                .format(k, step+1, len(val_dataloader),
                                        data_time = data_time, batch_time=batch_time,
                                        loss = losses, top1_acc = top1, top5_acc = top5
                                        )
                                )
                print(print_string)
        print_string = ('Testing Results: loss {loss.avg:.5f}, Top-1 {top1.avg:.3f}, Top-5 {top5.avg:.3f}'
                        .format(loss=losses, top1=top1, top5=top5)
                        )
        print(print_string)
    # 计算一个epoch的accuray、recall、precision、AUC
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    total_f1 = test_f1.compute()
    
    acc = total_acc.item()
    recall = total_recall.item()
    precision = total_precision.item()
    auc = total_auc.item()
    f1 = total_f1.item()

    print(f"Avg loss: {losses.avg:>8f}, "
                f"acc: {(100 * total_acc):>0.2f}%, "
                f"recall: {(100 * total_recall):>0.2f}%, "
                f"precision: {(100 * total_precision):>8f}, "
                f"auc: {(100 * total_auc.item()):>0.2f}%, "
                f"f1: {(100 * total_f1.item()):>0.2f}%, ")


    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_acc_epoch', total_acc, epoch)
    writer.add_scalar('val_recall_epoch', total_recall, epoch)
    writer.add_scalar('val_precision_epoch', total_precision, epoch)
    writer.add_scalar('val_auc_epoch', total_auc, epoch)
    writer.add_scalar('val_f1_epoch', total_f1, epoch)
     # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    test_auc.reset()
    test_f1.reset()

    model.train()
    
    return losses.avg, top1.avg, acc, recall, precision, auc, f1

def get_k_fold_data(k, k1, image_dir):
    # 返回第i折交叉验证时所需要的训练和验证数据
    # k1：当前折
    assert k > 1##K折交叉验证K大于1
    file = open(image_dir, 'r', encoding='utf-8',newline="")
    reader = csv.reader(file)
    imgs_ls = []
    for line in reader:
        imgs_ls.append(line)
    #print(len(imgs_ls))
    file.close()
    random.shuffle(imgs_ls)
    avg = len(imgs_ls) // k
    # w模式写入时会先将文件原内容清空，再写入新内容。a模式不会清空文件原内容，而是把新内容追加在原内容之后。
    f1 = open('annotation/train_{}.txt'.format(k1), 'w',newline='')
    f2 = open('annotation/val_{}.txt'.format(k1), 'w',newline='')
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)
    for i, row in enumerate(imgs_ls):
        #print(row)
        if (i // avg) == k1:
            writer2.writerow(row)
        else:
            writer1.writerow(row)
    f1.close()
    f2.close()


def k_fold(k,image_dir,num_epochs,batch_size, clip_len,  learning_rate, lr_steps, frame_path, num_workers):
    all_acc = []
    all_precision = []
    all_recall = []
    all_auc = []
    all_f1 = []
    
    for i in range(k): 
        seed_torch()
        # 指标
        best_top1 = 0.
        best_acc = 0.
        best_recall = 0.
        best_precision = 0.
        best_auc = 0.
        best_f1 = 0.
        
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

        # get_k_fold_data(k, i, image_dir)
        
        # train_k = 'annotation/train_{}.txt'.format(i)
        # test_k = 'annotation/val_{}.txt'.format(i)
        train_k = 'annotation/avg_K5/{}_train.txt'.format(i)
        test_k = 'annotation/avg_K5/{}_val.txt'.format(i)
        
        print("load dataset")
        train_data = dataset.dataset_video(is_train=True, frame_path=frame_path, root=train_k, clip_len=args.clip_len)
        test_data = dataset.dataset_video(is_train=False, frame_path=frame_path, root=test_k, clip_len=args.clip_len)

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        
        # defining model
        print("load model")
        model = TSN_model.TSN(args.num_classes, args.clip_len, 'RGB', 
                                is_shift = args.is_shift,
                                partial_bn = args.npb,
                                base_model=args.base_model, 
                                shift_div = args.shift_div, 
                                dropout=args.dropout, 
                                img_feature_dim = 224,
                                pretrain=args.pretrain, # 'imagenet' or False
                                consensus_type='avg',
                                fc_lr5 = True,
                                print_spec = False,
                                module=args.module)
        print(model)

        if args.pretrained is not None:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            try:
                model_dict = model.module.state_dict()
            except AttributeError:
                model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and 'fc' not in k}
            print("load pretrained model {}".format(args.pretrained))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if args.recover_from_checkpoint is not None:
            checkpoint = torch.load(args.recover_from_checkpoint, map_location='cpu')
            try:
                model_dict = model.module.state_dict()
            except AttributeError:
                model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            print("recover from checkpoint {}".format(args.recover_from_checkpoint))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        # 损失函数
        criterion = nn.CrossEntropyLoss().to(device)
        # potimizer
        # 返回学习率调整策略
        policies = model.get_optim_policies()
        model = nn.DataParallel(model)  # multi-Gpu
        model = model.to(device)

        for param_group in policies:
            param_group['lr'] = args.lr * param_group['lr_mult']
            param_group['weight_decay'] = args.weight_decay * param_group['decay_mult']
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        optimizer = optim.SGD(policies, momentum=args.momentum)
    
        # tensorboard, tensorboard --logdir 'tf_logs'
        logdir = os.path.join('tf_logs/', '{}'.format(args.save_path), '{}_{}'.format(args.base_model, args.clip_len), cur_time)
        if not os.path.exists(logdir):
            os.makedirs(logdir)    
        writer = SummaryWriter(log_dir=logdir)
        model_save_dir = os.path.join('checkpoint/', '{}'.format(args.save_path), '{}_{}'.format(args.base_model, args.clip_len), cur_time, '{}_{}'.format(args.base_model, args.clip_len))
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        scaler = GradScaler()
        # start training
        for epoch in trange(num_epochs):
            # acc, recall, precision, auc, f1 =  train_epoch(model, train_loader, epoch, loss, optimizer, writer)  
            train_epoch(i, model, train_loader, epoch, criterion, optimizer, writer, scaler)

            if epoch % 1 == 0:
                val_loss, val_top1, val_acc, val_recall, val_precision, val_auc, val_f1 = validation(i, model, test_loader, epoch, criterion, writer)
                if val_acc > best_acc:
                    checkpoint = os.path.join(model_save_dir,
                                            "clip_len_" + str(clip_len) + "_"+ str(i) 
                                            + "_checkpoint" + ".pth.tar")
                    utils.save_checkpoint(model, optimizer, checkpoint)
                    best_top1 = val_top1
                    best_acc = val_acc
                    loss = val_loss
                    best_recall = val_recall
                    best_precision = val_precision
                    best_auc = val_auc
                    best_f1 = val_f1
                print('fold: [{}], Best Top-1: {:.5f}, acc: {:.5f}, recall: {:.5f}, precision: {:.5f}, auc: {:.5f}, f1: {:.5f}, loss: {:.5f}'
                        .format(i, best_top1, best_acc, best_recall, best_precision, best_auc, best_f1, loss))
                # # 最好的评价指标存入csv
                # f = open('result/acc_result.csv','w')
                # csv_write = csv.writer(f)
                # csv_write.writerow(['fold','best top-1','acc','recall','precision','auc','f1'])
                # csv_write.writerow([str(i),str(best_top1),str(best_acc),str(best_recall),str(best_precision),str(best_auc),str(best_f1)])
                # f.close()
            utils.adjust_learning_rate(learning_rate, optimizer, epoch, lr_steps)
        all_acc.append(best_acc)
        all_auc.append(best_auc)
        all_precision.append(best_precision)
        all_recall.append(best_recall)
        all_f1.append(best_f1)
        writer.close

    return all_acc, all_recall, all_precision, all_auc, all_f1

def main():

    # k折交叉验证
    all_acc, all_recall, all_precision, all_auc, all_f1 = k_fold(k=args.k, 
                                    image_dir=args.image_dir, num_epochs=args.epochs, batch_size=args.batch_size, 
                                    clip_len=args.clip_len, 
                                    learning_rate=args.lr, lr_steps=args.lr_steps,
                                    frame_path=args.frame_path,
                                    num_workers=args.num_workers)
    
    for i in range(args.k):
        print('fold: {}, all_acc: {}, all_recall: {}, all_precision: {}, all_auc: {}, all_f1: {}'.format(str(i),all_acc[i],all_recall[i],all_precision[i],all_auc[i],all_f1[i]))
    print('avg_acc: {}, avg_recall: {}, avg_precision: {}, avg_auc: {}, avg_f1: {}'.format(np.mean(all_acc),np.mean(all_recall),np.mean(all_precision),np.mean(all_auc),np.mean(all_f1)))
    result_save_dir = 'result/{}'.format(args.save_path)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    f = open('result/{}/{}_{}.csv'.format(args.save_path, args.base_model, args.clip_len),'w')
    # f = open(result_save_dir,'w')
    csv_write = csv.writer(f)
    csv_write.writerow(['fold','acc','recall','precision','auc','f1'])
    for i in range(args.k):
        csv_write.writerow([str(i),all_acc[i],all_recall[i],all_precision[i],all_auc[i],all_f1[i]])
    csv_write.writerow(['Avg', np.mean(all_acc),np.mean(all_recall),np.mean(all_precision),np.mean(all_auc),np.mean(all_f1)])
    f.close()
    
       

if __name__ == '__main__':
    main()