import random

import numpy as np
import os, sys, time, datetime
from os.path import expanduser
import argparse

import subprocess

import torch.random

git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
git_branch = subprocess.Popen("git symbolic-ref --short -q HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)


parser = argparse.ArgumentParser(description="argument for GraphSAINT training")
parser.add_argument("--num_cpu_core", default=4, type=int, help="Number of CPU cores for parallel sampling")
parser.add_argument("--log_device_placement", default=False, action="store_true", help="Whether to log device placement")
parser.add_argument("--data_prefix", required=True, type=str, help="prefix identifying training data")
parser.add_argument("--dir_log", default="./test", type=str, help="base directory for logging and saving embeddings")
parser.add_argument("--gpu", default="0", type=str, help="which GPU to use")
parser.add_argument("--eval_train_every", default=1, type=int, help="How often to evaluate training subgraph accuracy")
parser.add_argument("--train_config", required=True, type=str, help="path to the configuration of training (*.yml)")
parser.add_argument("--dtype", default="s", type=str, help="d for double, s for single precision floating point")
parser.add_argument("--timeline", default=False, action="store_true", help="to save timeline.json or not")
parser.add_argument("--tensorboard", default=False, action="store_true", help="to save data to tensorboard or not")
parser.add_argument("--dualGPU", default=False, action="store_true", help="whether to distribute the model to two GPUs")
parser.add_argument("--cpu_eval", default=False, action="store_true", help="whether to use CPU to do evaluation")
parser.add_argument("--saved_model_path", default="", type=str, help="path to pretrained model file")
parser.add_argument("--repeat_time", default=10, type=int)
parser.add_argument("--sentence_embed", default="cnn", type=str)
parser.add_argument("--hidden_dim", type=int, default=-1)
parser.add_argument("--no_graph", default=False, action="store_true", )
parser.add_argument("--use_sam", default=False, action="store_true", help="whether to use SAM optimizer")
parser.add_argument('--sam_rho', type=float, default=0.05)
parser.add_argument('--sam_min_rho', type=float, default=0.05)
parser.add_argument('--sam_max_rho', type=float, default=0.8)
parser.add_argument('--sam_rho_schedule', default='none', choices=('none', 'linear', 'step'))
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy')
parser.add_argument('--FADRW_epoch', default=50, type=int, help='FADRW work epoch')
parser.add_argument('--FADRW_gamma', default=0.3, type=float, help='FADRW gamma')            
parser.add_argument('--FADRW_tau', default=1, type=float, help='FADRW tau')
args_global = parser.parse_args()


NUM_PAR_SAMPLER = args_global.num_cpu_core
SAMPLES_PER_PROC = -(-200 // NUM_PAR_SAMPLER)

EVAL_VAL_EVERY_EP = 1


gpu_selected = args_global.gpu
if gpu_selected == '-1234':
    gpu_stat = subprocess.Popen("nvidia-smi", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
    gpu_avail = set([str(i) for i in range(8)])
    for line in gpu_stat.split('\n'):
        if 'python' in line:
            if line.split()[1] in gpu_avail:
                gpu_avail.remove(line.split()[1])
            if len(gpu_avail) == 0:
                gpu_selected = -2
            else:
                gpu_selected = sorted(list(gpu_avail))[0]
    if gpu_selected == -1:
        gpu_selected = '0'
    args_global.gpu = int(gpu_selected)
if str(gpu_selected).startswith('nvlink'):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_selected).split('nvlink')[1]
elif int(gpu_selected) >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_selected)
    GPU_MEM_FRACTION = 0.8
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
args_global.gpu = int(args_global.gpu)


f_mean = lambda l: sum(l)/len(l)

DTYPE = "float32" if args_global.dtype == 's' else "float64"
