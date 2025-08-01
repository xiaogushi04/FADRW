import random  # 导入random模块，用于设置随机种子

import numpy as np  # 导入numpy模块，用于数值计算
import os,sys,time,datetime  # 导入os、sys、time、datetime模块，分别用于文件操作、系统操作、时间处理
from os.path import expanduser  # 导入expanduser函数，用于处理用户目录
import argparse  # 导入argparse模块，用于命令行参数解析

import subprocess  # 导入subprocess模块，用于执行系统命令

import torch.random  # 导入PyTorch的随机模块

# 获取当前git仓库的commit短哈希值
git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]  # 获取当前git提交的短哈希
# 获取当前git分支名
git_branch = subprocess.Popen("git symbolic-ref --short -q HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]  # 获取当前git分支名

# 获取当前时间戳，并格式化为字符串
timestamp = time.time()  # 获取当前时间戳
# 格式化时间戳为'年-月-日 时-分-秒'格式
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')

# 设置随机种子，保证实验可复现
seed = 42  # 随机种子
np.random.seed(seed)  # 设置numpy的随机种子
torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
torch.cuda.manual_seed_all(seed)  # 设置PyTorch的所有GPU随机种子
random.seed(seed)  # 设置python的random随机种子
#tf.set_random_seed(seed)  # TensorFlow的随机种子（已注释）


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="argument for GraphSAINT training")  # 创建ArgumentParser对象，描述信息为GraphSAINT训练参数
parser.add_argument("--num_cpu_core",default=4,type=int,help="Number of CPU cores for parallel sampling")  # 并行采样用的CPU核心数
parser.add_argument("--log_device_placement",default=False,action="store_true",help="Whether to log device placement")  # 是否记录设备分配日志
parser.add_argument("--data_prefix",required=True,type=str,help="prefix identifying training data")  # 训练数据前缀，必填
parser.add_argument("--dir_log",default="./test",type=str,help="base directory for logging and saving embeddings")  # 日志和embedding保存目录
parser.add_argument("--gpu",default="0",type=str,help="which GPU to use")  # 指定使用哪个GPU，默认自动选择
parser.add_argument("--eval_train_every",default=1,type=int,help="How often to evaluate training subgraph accuracy")  # 训练时每多少步评估一次
parser.add_argument("--train_config",required=True,type=str,help="path to the configuration of training (*.yml)")  # 训练配置文件路径，必填
parser.add_argument("--dtype", default="s", type=str, help="d for double, s for single precision floating point")  # 数据类型，d为双精度，s为单精度
parser.add_argument("--timeline", default=False, action="store_true",help="to save timeline.json or not")  # 是否保存timeline.json
parser.add_argument("--tensorboard",default=False,action="store_true",help="to save data to tensorboard or not")  # 是否保存到tensorboard
parser.add_argument("--dualGPU",default=False,action="store_true",help="whether to distribute the model to two GPUs")  # 是否使用双GPU
parser.add_argument("--cpu_eval",default=False,action="store_true",help="whether to use CPU to do evaluation")  # 是否在CPU上评估
parser.add_argument("--saved_model_path",default="",type=str,help="path to pretrained model file")  # 预训练模型路径
parser.add_argument("--repeat_time", default=10, type=int)  # 实验重复次数
parser.add_argument("--sentence_embed", default="cnn", type=str)  # 句子嵌入方式
parser.add_argument("--hidden_dim", type=int, default=-1)  # 隐藏层维度
parser.add_argument("--no_graph", default=False,action="store_true",)  # 是否不使用图结构
parser.add_argument("--use_sam", default=False, action="store_true", help="whether to use SAM optimizer")  #是否使用sam
parser.add_argument('--sam_rho', type=float, default=0.05)  # SAM优化器的rho参数
parser.add_argument('--sam_min_rho', type=float, default=0.05)  # SAM优化器的最小rho值
parser.add_argument('--sam_max_rho', type=float, default=0.8)  # SAM优化器的最大rho值
parser.add_argument('--sam_rho_schedule', default='none', choices=('none','linear', 'step'))  # rho调度策略
#parser.add_argument('--lr', type=float, default=0.001)  # 学习率
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')  # 损失类型
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy')  # 数据采样策略
parser.add_argument('--DRW_epoch', default=50, type=int, help='DRW work epoch')
parser.add_argument('--VSLoss_gamma', default=0.3,type=float, help='VSLoss gamma')
parser.add_argument('--VSLoss_tau', default=1,type=float, help='VSLoss tau')
args_global = parser.parse_args()  # 解析命令行参数，结果保存在args_global对象中

# args_global = parser.parse_args(["--data_prefix=./data/reddit-ac-1-onlyends_with_isolated_bi_10percent_hop1", "--gpu=0","--train_config=./train_config/gat_192_8_with_graph.yml", "--repeat_time=1",
#                                ])
# print(args_global.hidden_dim)

# 采样相关全局变量
NUM_PAR_SAMPLER = args_global.num_cpu_core  # 并行采样的CPU核心数
SAMPLES_PER_PROC = -(-200 // NUM_PAR_SAMPLER) # round up division，每个进程采样的样本数，向上取整

EVAL_VAL_EVERY_EP = 1       # 每多少个epoch在验证集上评估一次


# 自动选择可用的NVIDIA GPU
gpu_selected = args_global.gpu  # 获取命令行参数指定的GPU
if gpu_selected == '-1234':  # 如果为默认值，自动检测可用GPU
    # 通过nvidia-smi命令检测可用GPU
    gpu_stat = subprocess.Popen("nvidia-smi",shell=True,stdout=subprocess.PIPE,universal_newlines=True).communicate()[0]  # 获取nvidia-smi输出
    gpu_avail = set([str(i) for i in range(8)])  # 假设最多8块GPU，编号0-7
    for line in gpu_stat.split('\n'):
        if 'python' in line:  # 如果该行有python进程
            if line.split()[1] in gpu_avail:
                gpu_avail.remove(line.split()[1])  # 移除已被占用的GPU
            if len(gpu_avail) == 0:
                gpu_selected = -2  # 没有可用GPU
            else:
                gpu_selected = sorted(list(gpu_avail))[0]  # 选择最小编号的可用GPU
    if gpu_selected == -1:
        gpu_selected = '0'  # 如果没有检测到，默认用0号GPU
    args_global.gpu = int(gpu_selected)  # 更新全局参数
if str(gpu_selected).startswith('nvlink'):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected).split('nvlink')[1]  # 设置可见GPU为nvlink后面的编号
elif int(gpu_selected) >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # 设置CUDA设备顺序
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected)  # 设置可见GPU编号
    GPU_MEM_FRACTION = 0.8  # 设置GPU内存使用比例
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # 不使用GPU
args_global.gpu = int(args_global.gpu)  # 确保gpu参数为int类型

# 全局函数和变量

f_mean = lambda l: sum(l)/len(l)  # 求列表均值的lambda函数

DTYPE = "float32" if args_global.dtype=='s' else "float64"      # NOTE: currently not supporting float64 yet  # 根据参数设置数据类型，默认float32
