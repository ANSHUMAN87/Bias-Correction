import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(dir_path, os.path.pardir)
sys.path.append(root_dir)
import argparse
import time
import logging
import random
import shutil
import time
import collections
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.meters import AverageMeter, accuracy
from quantization.inference_quantization_manager import QuantizationManagerInference as QM
from utils.absorb_bn import search_absorbe_bn
import numpy as np
from pathlib import Path
import mlflow


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')

args = parser.parse_args()

class InferenceModel:
    def __init__(self):
        global args

        if 'cuda' in args.device and torch.cuda.is_available():
            torch.cuda.set_device(args.device_ids[0])
            cudnn.benchmark = True
        else:
            args.device_ids = None

        # create model
        print("=> using pre-trained model '{}'".format(args.arch))

        self.model = models.__dict__[args.arch](pretrained=True)

        # BatchNorm folding
        print("Perform BN folding")
        search_absorbe_bn(self.model)

        self.model.to(args.device)

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(args.device)

        cudnn.benchmark = True

        # Data loading code
        valdir = os.path.join(args.data, '')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        resize = 256 if args.arch != 'inception_v3' else 299
        crop_size = 224 if args.arch != 'inception_v3' else 299
        tfs = [
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose(tfs)),
            batch_size=args.batch_size, shuffle=(True),
            num_workers=args.workers, pin_memory=True)

    def run(self):
        val_loss, val_prec1, val_prec5 = validate(self.val_loader, self.model, self.criterion)
        if mlflow.active_run() is not None:
            mlflow.log_metric('top1', val_prec1)
            mlflow.log_metric('top5', val_prec5)
            mlflow.log_metric('loss', val_loss)
        return val_loss, val_prec1, val_prec5



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
      
    '''print("Validate begin")
    for n, m in self.model.named_modules():
            print(m)'''

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(args.device)
            target = target.to(args.device)
            output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(float(prec1), input.size(0))
            top5.update(float(prec5), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            #return

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    with QM():
        im = InferenceModel()
        im.run()
