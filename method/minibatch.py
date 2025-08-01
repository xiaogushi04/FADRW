from globals import *  # 导入全局变量和参数
from utils import *  # 导入工具函数
from graph_samplers import *  # 导入图采样器相关函数
from norm_aggr import *  # 导入归一化聚合相关函数
import torch  # 导入PyTorch
import scipy.sparse as sp  # 导入scipy稀疏矩阵库

import numpy as np  # 导入numpy
import time  # 导入time模块


def _coo_scipy2torch(adj):  # 将scipy的COO稀疏矩阵转为PyTorch稀疏张量
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data  # 获取非零元素的值
    indices = np.vstack((adj.row, adj.col))  # 获取非零元素的行列索引
    i = torch.LongTensor(indices)  # 转为LongTensor
    v = torch.FloatTensor(values)  # 转为FloatTensor
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))  # 构造PyTorch稀疏张量


class Minibatch:
    """
    Provides minibatches for the trainer or evaluator. This class is responsible for
    calling the proper graph sampler and estimating normalization coefficients.
    """
    def __init__(self, adj_full_norm, adj_train, role, train_params, cpu_eval=False):
        """
        Inputs:
            adj_full_norm       scipy CSR, adj matrix for the full graph (row-normalized)
            adj_train           scipy CSR, adj matrix for the traing graph. Since we are
                                under transductive setting, for any edge in this adj,
                                both end points must be training nodes.
            role                dict, key 'tr' -> list of training node IDs;
                                      key 'va' -> list of validation node IDs;
                                      key 'te' -> list of test node IDs.
            train_params        dict, additional parameters related to training. e.g.,
                                how many subgraphs we want to get to estimate the norm
                                coefficients.
            cpu_eval            bool, whether or not we want to run full-batch evaluation
                                on the CPU.

        Outputs:
            None
        """
        self.use_cuda = (args_global.gpu >= 0)  # 是否使用GPU
        if cpu_eval:
            self.use_cuda=False  # 评估时强制用CPU

        self.node_train = np.array(role['tr'])  # 训练集节点ID
        self.node_val = np.array(role['va'])  # 验证集节点ID
        self.node_test = np.array(role['te'])  # 测试集节点ID

        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())  # 全图归一化邻接矩阵转为PyTorch稀疏张量
        self.adj_train = adj_train  # 训练子图邻接矩阵
        # -----------------------
        # sanity check (optional)
        # -----------------------
        #for role_set in [self.node_val, self.node_test]:
        #    for v in role_set:
        #        assert self.adj_train.indptr[v+1] == self.adj_train.indptr[v]
        #_adj_train_T = sp.csr_matrix.tocsc(self.adj_train)
        #assert np.abs(_adj_train_T.indices - self.adj_train.indices).sum() == 0
        #assert np.abs(_adj_train_T.indptr - self.adj_train.indptr).sum() == 0
        #_adj_full_T = sp.csr_matrix.tocsc(adj_full_norm)
        #assert np.abs(_adj_full_T.indices - adj_full_norm.indices).sum() == 0
        #assert np.abs(_adj_full_T.indptr - adj_full_norm.indptr).sum() == 0
        #printf("SANITY CHECK PASSED", style="yellow")
        if self.use_cuda:
            # now i put everything on GPU. Ideally, full graph adj/feat
            # should be optionally placed on CPU
            self.adj_full_norm = self.adj_full_norm.cuda()  # 邻接矩阵转到GPU

        # below: book-keeping for mini-batch
        self.node_subgraph = None  # 当前子图节点
        self.batch_num = -1  # 当前batch编号

        self.method_sample = None  # 当前采样方法
        self.subgraphs_remaining_indptr = []  # 剩余子图的indptr
        self.subgraphs_remaining_indices = []  # 剩余子图的indices
        self.subgraphs_remaining_data = []  # 剩余子图的data
        self.subgraphs_remaining_nodes = []  # 剩余子图的节点
        self.subgraphs_remaining_edge_index = []  # 剩余子图的边索引

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])  # 训练集损失归一化系数
        # norm_loss_test is used in full batch evaluation (without sampling).
        # so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])  # 测试/验证集损失归一化系数
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)  # 总节点数
        self.norm_loss_test[self.node_train] = 1. / _denom  # 训练集节点归一化
        self.norm_loss_test[self.node_val] = 1. / _denom  # 验证集节点归一化
        self.norm_loss_test[self.node_test] = 1. / _denom  # 测试集节点归一化
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))  # 转为Tensor
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.cuda()  # 转到GPU
        self.norm_aggr_train = np.zeros(self.adj_train.size)  # 训练集聚合归一化系数

        self.sample_coverage = train_params['sample_coverage']  # 采样覆盖率
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()  # 训练集节点度数

    def set_sampler(self, train_phases):
        """
        Pick the proper graph sampler. Run the warm-up phase to estimate
        loss / aggregation normalization coefficients.

        Inputs:
            train_phases       dict, config / params for the graph sampler

        Outputs:
            None
        """
        self.subgraphs_remaining_indptr = []  # 清空子图缓存
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']  # 采样方法
        if self.method_sample == 'mrw':  # 多阶随机游走采样
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000      # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']  # 子图大小预算
            self.graph_sampler = mrw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                train_phases['size_frontier'],
                _deg_clip,
            )
        elif self.method_sample == 'rw':  # 随机游走采样
            self.size_subg_budget = train_phases['num_root'] * train_phases['depth']
            self.graph_sampler = rw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                int(train_phases['num_root']),
                int(train_phases['depth']),
            )
        elif self.method_sample == 'edge':  # 边采样
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = edge_sampling(
                self.adj_train,
                self.node_train,
                train_phases['size_subg_edge'],
            )
        elif self.method_sample == 'node':  # 节点采样
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == 'full_batch':  # 全图采样
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == "vanilla_node_python":  # 纯Python节点采样
            self.size_subg_budget = train_phases["size_subgraph"]
            self.graph_sampler = NodeSamplingVanillaPython(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        else:
            raise NotImplementedError  # 未实现的采样方法

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])  # 重新初始化损失归一化系数
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)  # 重新初始化聚合归一化系数

        # -------------------------------------------------------------
        # BELOW: estimation of loss / aggregation normalization factors
        # -------------------------------------------------------------
        # For some special sampler, no need to estimate norm factors, we can calculate
        # the node / edge probabilities directly.
        # However, for integrity of the framework, we follow the same procedure
        # for all samplers:
        #   1. sample enough number of subgraphs
        #   2. update the counter for each node / edge in the training graph
        #   3. estimate norm factor alpha and lambda
        tot_sampled_nodes = 0  # 已采样节点数
        while True:
            self.par_graph_sample('train')  # 并行采样子图
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])  # 统计采样节点数
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break  # 达到采样覆盖率则停止
        print()
        num_subg = len(self.subgraphs_remaining_nodes)  # 子图数量
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1  # 边计数
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1  # 节点计数
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0  # 验证/测试集节点不应被采样
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v + 1]
            val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)  # 计算归一化系数
            val[np.isnan(val)] = 0.1  # NaN设为0.1
            self.norm_aggr_train[i_s : i_e] = val  # 更新聚合归一化系数
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1  # 未采样节点设为0.1
        self.norm_loss_train[self.node_val] = 0  # 验证集节点归一化系数为0
        self.norm_loss_train[self.node_test] = 0  # 测试集节点归一化系数为0
        self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size  # 训练集节点归一化
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))  # 转为Tensor
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.cuda()  # 转到GPU

    def par_graph_sample(self,phase):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        t0 = time.time()  # 记录开始时间
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)  # 调用采样器采样
        t1 = time.time()  # 记录结束时间
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end="\r")  # 打印采样耗时
        self.subgraphs_remaining_indptr.extend(_indptr)  # 缓存采样结果
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def one_batch(self, mode='train'):
        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds
        to one subgraph of the training graph. In the 'val' or 'test' mode, one batch
        corresponds to the full graph (i.e., full-batch rather than minibatch evaluation
        for validation / test sets).

        Inputs:
            mode                str, can be 'train', 'val', 'test' or 'valtest'

        Outputs:
            node_subgraph       np array, IDs of the subgraph / full graph nodes
            adj                 scipy CSR, adj matrix of the subgraph / full graph
            norm_loss           np array, loss normalization coefficients. In 'val' or
                                'test' modes, we don't need to normalize, and so the values
                                in this array are all 1.
        """
        if mode in ['val','test','valtest']:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])  # 全图节点
            adj = self.adj_full_norm  # 全图邻接矩阵
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')  # 采样新子图
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()  # 取一个子图节点
            self.size_subgraph = len(self.node_subgraph)  # 子图节点数
            adj = sp.csr_matrix(
                (
                    self.subgraphs_remaining_data.pop(),
                    self.subgraphs_remaining_indices.pop(),
                    self.subgraphs_remaining_indptr.pop()),
                    shape=(self.size_subgraph,self.size_subgraph,
                )
            )  # 构造子图邻接矩阵
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()  # 边索引
            #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
            norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=args_global.num_cpu_core)  # 归一化聚合
            # adj.data[:] = self.norm_aggr_train[adj_edge_index][:]      # this line is interchangable with the above line
            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])  # 邻接矩阵归一化
            adj = _coo_scipy2torch(adj.tocoo())  # 转为PyTorch稀疏张量
            if self.use_cuda:
                adj = adj.cuda()  # 转到GPU
            self.batch_num += 1  # batch编号加一
        norm_loss = self.norm_loss_test if mode in ['val','test', 'valtest'] else self.norm_loss_train  # 损失归一化系数
        norm_loss = norm_loss[self.node_subgraph]  # 取当前子图的归一化系数
        return self.node_subgraph, adj, norm_loss  # 返回子图节点、邻接矩阵、归一化系数


    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))  # 计算训练batch数

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)  # 打乱训练节点顺序
        self.batch_num = -1  # 重置batch编号

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]  # 判断是否遍历完所有batch
