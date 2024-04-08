import argparse
from utils.config_utils import setup_seed

is_sbatch = 1
if is_sbatch:
    data_pth = r'/share/home/202220143416/time_series_data/dc_detector/'
    logs_pth = r'/share/home/202220143416/TF_AD_models_logs/logs/'
else:
    data_pth = r'/home/zzj/time_series_data2/DCdetector_dataset/'
    logs_pth = r'/home/zzj/TF_AD_models_logs/logs/'


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training Argument Parser')
    # 数据路径
    parser.add_argument('--data_pth', type=str, default=data_pth, help='Path to the dataset')
    parser.add_argument('--data_name', type=str, default='WADI', help='dataset')
    parser.add_argument('--index', type=int, default=1, help='dataset')

    parser.add_argument('--model_id', type=str, default='none', help='model_id')

    # patch_size
    parser.add_argument('--patch_size', type=int, default=32, help='patch_size')
    # d_model
    parser.add_argument('--d_model', type=int, default=512, help='d_model')
    # ffn_scale
    parser.add_argument('--ffn_scale', type=int, default=4, help='ffn_scale')
    # n_head
    parser.add_argument('--n_head', type=int, default=8, help='n_head')
    # n_layers
    parser.add_argument('--n_layers', type=int, default=8, help='n_layers')
    # q_len
    parser.add_argument('--q_len', type=int, default=1000, help='q_len')

    # =========== 训练参数 =================
    # 训练周期
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    # 批次大小
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    # accumulate_grad_steps
    parser.add_argument('--accumulate_grad_steps', type=int, default=1, help='accumulate_grad_steps')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # 动量
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    # 打印频率
    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--test_freq', type=int, default=50, help='Test frequency')
    # AMP混合精度
    parser.add_argument('--use_amp', type=int, default=0, help='Use AMP')
    # 多进程
    parser.add_argument('--multiprocessing', type=int, default=0, help='Number of workers')
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        choices=[-1,1],
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training",choices=[-1,0]
    )
    parser.add_argument(
        "--dist_url",
        default="tcp://127.0.0.1:23437",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    # workers
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')


    # 恢复训练
    parser.add_argument('--resume', type=int, default=0, help='Path to saved model checkpoint for resuming training')
    # 模型保存路径
    parser.add_argument('--logs_pth', type=str, default=logs_pth, help='Path to save the logs')
    parser.add_argument('--save_pth', type=str, default='saved_models/', help='Path to save the trained models')
    # 随机种子
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    # 优化器选择
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use (SGD, Adam, etc.)')
    # 是否使用分布式训练
    parser.add_argument('--distributed', action='store_true', help='Use DistributedDataParallel for training')

    # gpu
    parser.add_argument('--gpu', type=int, default=None, help='gpu')

    ## model

    # ar
    parser.add_argument('--ar', type=float, default=0.1, help='ar')
    # win_size
    parser.add_argument('--win_size', type=int, default=2048, help='win_size')
    # step
    parser.add_argument('--step', type=int, default=1, help='step')

    # noise level
    parser.add_argument('--noise_level', type=float, default=0.1)

    # warmup_steps
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--warmup_max_ratio', type=float, default=0.5)

    return parser



from models_v2.model2_lazy import ContAD_wo_ci
from trainer_v2.trainer2 import ContAD_Trainer

import torch
import os

import sys
sys.path.append('../')

from data.data_loader2 import data_name2nc, get_loader_segment


import torch.multiprocessing as mp
import torch.distributed as dist
import warnings


def main():

    args_parser = get_arg_parser()
    args = args_parser.parse_args()

    nc = data_name2nc(args.data_name)
    args.nc = nc - 0


    if args.seed >= 0:
        setup_seed(args.seed)

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("=> Using multiprocessing distributed training")

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print("=> Using distributed training with {} GPUs".format(ngpus_per_node))

    print("==> Building model:")

   
    model = ContAD_wo_ci(c_dim=args.nc, seq_len=args.win_size, patch_size=args.patch_size, d_model=args.d_model, d_feed_foward_scale=args.ffn_scale, dropout=0.1, n_head=args.n_head, n_layers=args.n_layers, with_inter=0, with_intra=0, query_len=args.q_len) # test 32ps


    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu],broadcast_buffers=False,find_unused_parameters=True
                )
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False,find_unused_parameters=True)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    if args.gpu==-1:
        device = torch.device("cpu")

    elif torch.cuda.is_available():
        if args.gpu:
            device = torch.device("cuda:{}".format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    

    print(device)
    train_exp(model, args)



def train_exp(model, args):
    # print arg
    for k,v in vars(args).items():
        print(k,v)
    trainer = ContAD_Trainer(args, model, get_loader_segment)

    if args.resume == 0:
        trainer.train()
        trainer.test(0)
    elif args.resume == 1:
        trainer.test(1)
    elif args.resume == 2:
        trainer.train()
        trainer.test(0)


if __name__ == "__main__":
    main()
    

    

    
    
    
