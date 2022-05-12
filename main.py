import argparse
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from alexnet_pytorch import AlexNet
from utils import adjust_learning_rate
from utils import save_checkpoint
from utils import train
from utils import validate
from utils import Cutout


mixed_precision = True
try:
    from apex import amp
except:
    mixed_precision = False
    warnings.warn("Warning: Apex tool not install.")

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('data', metavar='DIR', default='data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    help='model architecture (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--opt_level', default="O1", type=str,
                    help="Choose which accuracy to train. (default: 'O1')")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_classes', type=int, default=10,
                    help="number of dataset category.")
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--augment',default="none",choices=['cutmix','cutout','mixup','none'])
parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha for mixup and cutmix')
parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=112,
                        help='length of the holes')
parser.add_argument('--cutmix_prob', type=float, default=0.5,
                        help='prob for cutmix')
best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print(args.data)
        print(f"Use GPU: {args.gpu} for training!")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'alexnet' in args.arch:
        if args.pretrained:
            model = AlexNet.from_pretrained(args.arch, args.num_classes)
            print(f"=> using pre-trained model '{args.arch}'")
        else:
            print(f"=> creating model '{args.arch}'")
            model = AlexNet.from_name(args.arch)
    else:
        if args.pretrained:
            print(f"=> using pre-trained model '{args.arch}'")
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print(f"=> creating model '{args.arch}'")
            model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available
        # GPUs
        if args.arch.startswith('alexnet'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train=transforms.Compose([
            # 剪裁成224
            transforms.Resize(224),
            # 中心对称
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    # cutout
    if args.augment == "cutout":
        transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        

    train_dataset = datasets.CIFAR100(
        root=args.data,
        download=True,
        train=True,
        transform=transform_train)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = datasets.CIFAR100(
        root=args.data,
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    #     top1 = validate(val_loader, model, criterion, args)
    #     with open('res.txt', 'w') as f:
    #         print(f"Acc@1: {top1}", file=f)
    #     return

    # visualization tensorboard
    # save path
    
    logdir = "results/" + args.augment
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir + "/logs")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc1,train_acc5,train_losses=train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1,acc5,losses = validate(val_loader, model, criterion, args)

        # plot tensorborad
        writer.add_scalar("train_acc_top1",train_acc1,epoch)
        writer.add_scalar("train_acc_top5",train_acc5,epoch)
        writer.add_scalar("train_loss",train_losses,epoch)
        writer.add_scalar("val_acc_top1",acc1,epoch)
        writer.add_scalar("val_acc_top5",acc5,epoch)
        writer.add_scalar("val_loss",losses,epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
            }, is_best, args.augment)
    writer.close()


if __name__ == '__main__':
    main()
