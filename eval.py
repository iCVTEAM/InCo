import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import time
from torchvision.models.resnet import resnet50
from scipy.stats.stats import ModeResult
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import tensorboard_logger as tb_logger
from dataset.transform_cfg import transforms_options, transforms_list
from dataset.mini_imagenet import ImageNet, MetaImageNet
from eval.meta_eval import meta_test
from transfer import FeatureExtractor
from pdb import set_trace
from moco.convnet import ConvNet
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', 
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='convnet')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--id', type=str, default='')

#few-shot larning
    # load pretrained model
#parser.add_argument('--model', type=str, default='convnet', choices=model_pool)
parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
parser.add_argument('--transform', type=str, default='B', choices=transforms_list)

    # meta setting
parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
parser.add_argument('--n_aug_support_samples', default=1, type=int,
                        help='The number of  samples for each meta test sample')
parser.add_argument('--data_root', type=str, default='data', metavar='N',
                        help='Root dataset')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='Number of workers for dataloader')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
opt = parser.parse_args()
   # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
opt.data_aug = True
opt.use_trainval = False
    #print(opt.dataset, hostname)
    
opt.data_root = '/media/space/WH/WH/fewshot-CAN/'

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
    
    args.tb_folder = 'Linear_eval/{}_tensorboard'.format(args.id)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
        
    ngpus_per_node = torch.cuda.device_count()
    #print('aaa',123)
    #main_worker(args.gpu, ngpus_per_node, args)  
    #print('test_acc: {:.4f}, test_std: {:.4f}'.format(test_acc, test_std))
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
    args.gpu = gpu
    #print('sdeffsaf:',args.gpu)    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
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
    print("=> creating model '{}'".format(args.arch))
    model = ConvNet
    model = model(num_classes=128)
    # freeze all layers but the last fc
    model_name = []
    for name, param in model.named_parameters():
        #print(name)
        model_name.append(name)
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
         
    if args.gpu==0:
        logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    else:  
        logger = None
          
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
  
            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            #set_trace()
            model = FeatureExtractor(model, ['avgpool'])
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            args.workers  =8
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    #model.cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = torch.utils.data.DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=True),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
    start = time.time()
    # print(model)
    test_acc, test_std = meta_test(model, meta_testloader, opt)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std,
                                                                    test_time))
if __name__ == '__main__':
    main()
