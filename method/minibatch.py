from globals import *  # ����ȫ�ֱ����Ͳ���
from utils import *  # ���빤�ߺ���
from graph_samplers import *  # ����ͼ��������غ���
from norm_aggr import *  # �����һ���ۺ���غ���
import torch  # ����PyTorch
import scipy.sparse as sp  # ����scipyϡ������

import numpy as np  # ����numpy
import time  # ����timeģ��


def _coo_scipy2torch(adj):  # ��scipy��COOϡ�����תΪPyTorchϡ������
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data  # ��ȡ����Ԫ�ص�ֵ
    indices = np.vstack((adj.row, adj.col))  # ��ȡ����Ԫ�ص���������
    i = torch.LongTensor(indices)  # תΪLongTensor
    v = torch.FloatTensor(values)  # תΪFloatTensor
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))  # ����PyTorchϡ������


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
        self.use_cuda = (args_global.gpu >= 0)  # �Ƿ�ʹ��GPU
        if cpu_eval:
            self.use_cuda=False  # ����ʱǿ����CPU

        self.node_train = np.array(role['tr'])  # ѵ�����ڵ�ID
        self.node_val = np.array(role['va'])  # ��֤���ڵ�ID
        self.node_test = np.array(role['te'])  # ���Լ��ڵ�ID

        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())  # ȫͼ��һ���ڽӾ���תΪPyTorchϡ������
        self.adj_train = adj_train  # ѵ����ͼ�ڽӾ���
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
            self.adj_full_norm = self.adj_full_norm.cuda()  # �ڽӾ���ת��GPU

        # below: book-keeping for mini-batch
        self.node_subgraph = None  # ��ǰ��ͼ�ڵ�
        self.batch_num = -1  # ��ǰbatch���

        self.method_sample = None  # ��ǰ��������
        self.subgraphs_remaining_indptr = []  # ʣ����ͼ��indptr
        self.subgraphs_remaining_indices = []  # ʣ����ͼ��indices
        self.subgraphs_remaining_data = []  # ʣ����ͼ��data
        self.subgraphs_remaining_nodes = []  # ʣ����ͼ�Ľڵ�
        self.subgraphs_remaining_edge_index = []  # ʣ����ͼ�ı�����

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])  # ѵ������ʧ��һ��ϵ��
        # norm_loss_test is used in full batch evaluation (without sampling).
        # so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])  # ����/��֤����ʧ��һ��ϵ��
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)  # �ܽڵ���
        self.norm_loss_test[self.node_train] = 1. / _denom  # ѵ�����ڵ��һ��
        self.norm_loss_test[self.node_val] = 1. / _denom  # ��֤���ڵ��һ��
        self.norm_loss_test[self.node_test] = 1. / _denom  # ���Լ��ڵ��һ��
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))  # תΪTensor
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.cuda()  # ת��GPU
        self.norm_aggr_train = np.zeros(self.adj_train.size)  # ѵ�����ۺϹ�һ��ϵ��

        self.sample_coverage = train_params['sample_coverage']  # ����������
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()  # ѵ�����ڵ����

    def set_sampler(self, train_phases):
        """
        Pick the proper graph sampler. Run the warm-up phase to estimate
        loss / aggregation normalization coefficients.

        Inputs:
            train_phases       dict, config / params for the graph sampler

        Outputs:
            None
        """
        self.subgraphs_remaining_indptr = []  # �����ͼ����
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']  # ��������
        if self.method_sample == 'mrw':  # ���������߲���
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000      # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']  # ��ͼ��СԤ��
            self.graph_sampler = mrw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                train_phases['size_frontier'],
                _deg_clip,
            )
        elif self.method_sample == 'rw':  # ������߲���
            self.size_subg_budget = train_phases['num_root'] * train_phases['depth']
            self.graph_sampler = rw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                int(train_phases['num_root']),
                int(train_phases['depth']),
            )
        elif self.method_sample == 'edge':  # �߲���
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = edge_sampling(
                self.adj_train,
                self.node_train,
                train_phases['size_subg_edge'],
            )
        elif self.method_sample == 'node':  # �ڵ����
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == 'full_batch':  # ȫͼ����
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == "vanilla_node_python":  # ��Python�ڵ����
            self.size_subg_budget = train_phases["size_subgraph"]
            self.graph_sampler = NodeSamplingVanillaPython(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        else:
            raise NotImplementedError  # δʵ�ֵĲ�������

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])  # ���³�ʼ����ʧ��һ��ϵ��
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)  # ���³�ʼ���ۺϹ�һ��ϵ��

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
        tot_sampled_nodes = 0  # �Ѳ����ڵ���
        while True:
            self.par_graph_sample('train')  # ���в�����ͼ
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])  # ͳ�Ʋ����ڵ���
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break  # �ﵽ������������ֹͣ
        print()
        num_subg = len(self.subgraphs_remaining_nodes)  # ��ͼ����
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1  # �߼���
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1  # �ڵ����
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0  # ��֤/���Լ��ڵ㲻Ӧ������
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v + 1]
            val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)  # �����һ��ϵ��
            val[np.isnan(val)] = 0.1  # NaN��Ϊ0.1
            self.norm_aggr_train[i_s : i_e] = val  # ���¾ۺϹ�һ��ϵ��
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1  # δ�����ڵ���Ϊ0.1
        self.norm_loss_train[self.node_val] = 0  # ��֤���ڵ��һ��ϵ��Ϊ0
        self.norm_loss_train[self.node_test] = 0  # ���Լ��ڵ��һ��ϵ��Ϊ0
        self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size  # ѵ�����ڵ��һ��
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))  # תΪTensor
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.cuda()  # ת��GPU

    def par_graph_sample(self,phase):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        t0 = time.time()  # ��¼��ʼʱ��
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)  # ���ò���������
        t1 = time.time()  # ��¼����ʱ��
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end="\r")  # ��ӡ������ʱ
        self.subgraphs_remaining_indptr.extend(_indptr)  # ����������
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
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])  # ȫͼ�ڵ�
            adj = self.adj_full_norm  # ȫͼ�ڽӾ���
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')  # ��������ͼ
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()  # ȡһ����ͼ�ڵ�
            self.size_subgraph = len(self.node_subgraph)  # ��ͼ�ڵ���
            adj = sp.csr_matrix(
                (
                    self.subgraphs_remaining_data.pop(),
                    self.subgraphs_remaining_indices.pop(),
                    self.subgraphs_remaining_indptr.pop()),
                    shape=(self.size_subgraph,self.size_subgraph,
                )
            )  # ������ͼ�ڽӾ���
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()  # ������
            #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
            norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=args_global.num_cpu_core)  # ��һ���ۺ�
            # adj.data[:] = self.norm_aggr_train[adj_edge_index][:]      # this line is interchangable with the above line
            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])  # �ڽӾ����һ��
            adj = _coo_scipy2torch(adj.tocoo())  # תΪPyTorchϡ������
            if self.use_cuda:
                adj = adj.cuda()  # ת��GPU
            self.batch_num += 1  # batch��ż�һ
        norm_loss = self.norm_loss_test if mode in ['val','test', 'valtest'] else self.norm_loss_train  # ��ʧ��һ��ϵ��
        norm_loss = norm_loss[self.node_subgraph]  # ȡ��ǰ��ͼ�Ĺ�һ��ϵ��
        return self.node_subgraph, adj, norm_loss  # ������ͼ�ڵ㡢�ڽӾ��󡢹�һ��ϵ��


    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))  # ����ѵ��batch��

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)  # ����ѵ���ڵ�˳��
        self.batch_num = -1  # ����batch���

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]  # �ж��Ƿ����������batch
