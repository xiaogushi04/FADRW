import torch  # 导入PyTorch主库
from torch import nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式API
from utils import *  # 导入自定义工具函数
import method.layers as layers  # 导入自定义的图神经网络层模块
import method.sam as sam  # 导入SAM模块
from globals import args_global
from method.losses import LDAMLoss, FocalLoss, VSLoss
import numpy as np  # 用于数值计算


class GraphSAINT(nn.Module):  # 定义GraphSAINT图神经网络主模型
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Build the multi-layer GNN architecture.

        Inputs:
            num_classes         int, number of classes a node can belong to
            arch_gcn            dict, config for each GNN layer
            train_params        dict, training hyperparameters (e.g., learning rate)
            feat_full           np array of shape N x f, where N is the total num of
                                nodes and f is the dimension for input node feature
            label_full          np array, for single-class classification, the shape
                                is N x 1 and for multi-class classification, the
                                shape is N x c (where c = num_classes)
            cpu_eval            bool, if True, will put the model on CPU.

        Outputs:
            None
        """
        super(GraphSAINT,self).__init__()  # 调用父类初始化
        self.vocab_size = np.max(feat_full)+1  # 词表大小，假设特征为词id
        self.mask = np.zeros_like(feat_full)  # 创建与特征同形状的全零mask
        self.mask[feat_full==0]=1  # 特征为0的位置mask为1
        self.mask = torch.from_numpy(self.mask.astype(np.bool))  # 转为PyTorch布尔张量


        # CNN sentence embedding

        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # dropout_prob = train_params["dropout"]
        # self.dropout = nn.Dropout(dropout_prob)

        # self.sentence_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1)
        self.use_cuda = (args_global.gpu >= 0)  # 是否使用GPU
        if cpu_eval:
            self.use_cuda=False  # 评估时强制用CPU
        if "attention" in arch_gcn:  # 如果配置中有attention
            if "gated_attention" in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls = layers.GatedAttentionAggregator  # 使用门控注意力聚合器
                    self.mulhead = int(arch_gcn['attention'])  # 多头数
            else:
                self.aggregator_cls = layers.AttentionAggregator  # 使用普通注意力聚合器
                self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls = layers.HighOrderAggregator  # 默认高阶聚合器
            self.mulhead = 1  # 单头
        self.num_layers = len(arch_gcn['arch'].split('-'))  # 层数
        self.weight_decay = train_params['weight_decay']  # 权重衰减
        self.dropout = train_params['dropout']  # dropout概率
        self.lr = train_params['lr']  # 学习率
        self.arch_gcn = arch_gcn  # 保存结构配置
        self.sigmoid_loss = (arch_gcn['loss'] == 'sigmoid')  # 是否用sigmoid损失
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))  # 全图特征转为Tensor
        self.label_full = torch.from_numpy(label_full.astype(np.float32))  # 全图标签转为Tensor
        self.sentence_embed_method = train_params["sentence_embed"]  # 句子嵌入方法
        self.sentence_embedding_dim = self.set_sentence_embedding(train_params["sentence_embed"])  # 设置句子嵌入层并获取输出维度
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()  # 特征转到GPU
            self.label_full = self.label_full.cuda()  # 标签转到GPU
            self.mask = self.mask.cuda()  # mask转到GPU
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(label_full.argmax(axis=1).astype(np.int64))  # 多分类时转为类别索引
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()  # 类别索引转到GPU
        self.num_classes = num_classes  # 类别数
        # _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
        #                 = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
                        = parse_layer_yml(arch_gcn, self.sentence_embedding_dim)  # 解析网络结构配置
        # _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
        #                 = parse_layer_yml(arch_gcn, filter_num*len(filter_size))
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()  # 设置卷积层索引
        self.set_dims(_dims)  # 设置各层输入输出维度

        self.loss = 0  # 初始化损失
        self.opt_op = None  # 优化器操作

        # build the model below
        self.num_params = 0  # 参数计数
        self.aggregators, num_param = self.get_aggregators()  # 获取聚合器层和参数数
        self.num_params += num_param  # 累加参数数
        self.conv_layers = nn.Sequential(*self.aggregators)  # 聚合器层组成的序列
        self.hidden_dim = train_params["hidden_dim"]  # 隐藏层维度
        self.no_graph = train_params["no_graph"]  # 是否不使用图结构
        self.dropout_layer = nn.Dropout(self.dropout)  # dropout层
        if self.hidden_dim == -1:
            if self.no_graph:
                print("NO GRAPH")
                self.classifier = layers.HighOrderAggregator(self.sentence_embedding_dim, self.num_classes, act='I', order=0, dropout=self.dropout, bias='bias')  # 只用句子嵌入做分类
                self.num_params += self.classifier.num_param
            else:
                # self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.num_classes,\
                #                      act='I', order=0, dropout=self.dropout, bias='bias')
                self.classifier = layers.HighOrderAggregator(self.dims_feat[-1]+self.sentence_embedding_dim, self.num_classes,\
                                     act='I', order=0, dropout=self.dropout, bias='bias')  # 拼接图特征和句子嵌入做分类
                # self.classifier2 = layers.HighOrderAggregator(self.sentence_embedding_dim,
                #                                              self.num_classes, act='I', order=0, dropout=self.dropout, bias='bias')
                # self.num_params += self.classifier.num_param + self.classifier2.num_param
                self.num_params += self.classifier.num_param  # 累加分类器参数数
        else:
            self.classifier_ = layers.HighOrderAggregator(self.dims_feat[-1]+self.sentence_embedding_dim, self.hidden_dim,\
                                act='relu', order=0, dropout=self.dropout, bias='norm-nn')  # 先拼接后接高阶聚合器
            self.classifier = nn.Linear(self.hidden_dim,self.num_classes)  # 再接线性层输出类别
            self.num_params += self.classifier_.num_param + self.num_classes*self.hidden_dim  # 累加参数数
        self.sentence_embed_norm = nn.BatchNorm1d(self.sentence_embedding_dim, eps=1e-9, track_running_stats=True)  # 句子嵌入归一化
        self.use_sam = args_global.use_sam
        if self.use_sam:
            print("USE SAM OPTIMIZER")
            self.rho = args_global.sam_rho
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # Adam优化器

    def set_dims(self, dims):
        """
        Set the feature dimension / weight dimension for each GNN or MLP layer.
        We will use the dimensions set here to initialize PyTorch layers.

        Inputs:
            dims        list, length of node feature for each hidden layer

        Outputs:
            None
        """
        self.dims_feat = [dims[0]] + [
            ((self.aggr_layer[l]=='concat') * self.order_layer[l] + 1) * dims[l+1]
            for l in range(len(dims) - 1)
        ]  # 计算每层的特征维度
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]  # 每层的权重维度

    def set_idx_conv(self):
        """
        Set the index of GNN layers for the full neural net. For example, if
        the full NN is having 1-0-1-0 arch (1-hop graph conv, followed by 0-hop
        MLP, ...). Then the layer indices will be 0, 2.
        """
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]  # 找出所有卷积层索引
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv  # 保存卷积层索引
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])  # 只保留order为1的层索引


    def cos_sim(self, x, is_training):
        if is_training:
            return nn.functional.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)  # 训练时直接算相似度
        B, L = x.shape
        sims = torch.zeros(size=(B,B))  # 初始化相似度矩阵
        for i in range(B):
            sims[i,:] = nn.functional.cosine_similarity(x, x[i].unsqueeze(0))  # 计算每一行与其他行的相似度
        return sims.to(x.device)  # 返回与输入同设备的相似度矩阵

    def top_sim(self, sims, topk=3):
        sims_, indices_ = sims.sort(descending=True)  # 对相似度排序
        B,_ = sims.shape
        indices = torch.zeros(size=(2,B*topk))  # 存储topk索引
        values = torch.zeros(size=(1,B*topk))  # 存储topk值
        for i,inds in enumerate(indices_):
            indices[0, i*topk:(i+1)*topk] = i*torch.ones(size=(1,topk))  # 行索引
            indices[1, i * topk:(i + 1) * topk] = inds[:topk]  # topk列索引
            values[0, i * topk:(i + 1) * topk] = sims[i][:topk]  # topk相似度值
        return torch.sparse_coo_tensor(indices,values.squeeze(0),size=(B,B))  # 返回稀疏相似度张量
        # return indices, values

    def set_sentence_embedding(self, method="cnn"):
        if method=="cnn":
            embed_size = 128  # 嵌入维度
            filter_size = [3, 4, 5]  # 卷积核尺寸
            filter_num = 128  # 每种卷积核数量
            self.embedding = nn.Embedding(self.vocab_size, embed_size)  # 词嵌入层
            self.cnn_list = nn.ModuleList()  # 卷积层列表
            for size in filter_size:
                self.cnn_list.append(nn.Conv1d(embed_size, filter_num, size))  # 添加不同尺寸的卷积层
            self.relu = nn.ReLU()  # 激活函数
            self.max_pool = nn.AdaptiveMaxPool1d(1)  # 最大池化层
            self.sentence_embed=self.cnn_embed  # 嵌入方法指向cnn_embed
            return len(filter_size) * filter_num  # 返回输出维度
        '''
        one layer FCN equals only pool
        '''
        if method == "maxpool":
            self.embed_size = 128  # 嵌入维度
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)  # 词嵌入层
            self.relu = nn.ReLU()  # 激活函数
            self.embed_pool = nn.AdaptiveMaxPool1d(1)  # 最大池化层
            self.sentence_embed=self.pool_embed  # 嵌入方法指向pool_embed
            return self.embed_size  # 返回输出维度
        if method == "avgpool":
            self.embed_size = 128  # 嵌入维度
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)  # 词嵌入层
            self.relu = nn.ReLU()  # 激活函数
            self.embed_pool = nn.AdaptiveAvgPool1d(1)  # 平均池化层
            self.sentence_embed = self.pool_embed  # 嵌入方法指向pool_embed
            return self.embed_size  # 返回输出维度
        if method == "rnn":
            self.embed_size = 128  # 嵌入维度
            hidden_dim = 64  # RNN隐藏层维度
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)  # 词嵌入层
            self.rnn = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)  # 双向LSTM
            self.sentence_embed=self.RNN_embed  # 嵌入方法指向RNN_embed
            return 2 * hidden_dim  # 返回输出维度
        if method == "lstm":
            self.embed_size = 128  # 嵌入维度
            hidden_dim = 64  # LSTM隐藏层维度
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)  # 词嵌入层
            self.rnn = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)  # 双向LSTM
            self.sentence_embed=self.LSTM_embed  # 嵌入方法指向LSTM_embed
            return 2 * hidden_dim * 2  # 返回输出维度
        if method == "lstmatt":
            self.embed_size = 128  # 嵌入维度
            hidden_dim = 64  # LSTM隐藏层维度
            self.LSTMATT_DP=nn.Dropout(self.dropout)  # dropout层
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)  # 词嵌入层
            self.lstm = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)  # 双向LSTM
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=1, dropout=self.dropout)  # 多头注意力
            self.sentence_embed = self.LSTMATT_embed  # 嵌入方法指向LSTMATT_embed
            self.relu = nn.ReLU()  # 激活函数
            self.embed_pool = nn.AdaptiveAvgPool1d(1)  # 平均池化层
            return hidden_dim*2  # 返回输出维度
        if method == "Transformer":
            self.embed_size = 128  # 嵌入维度
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)  # 词嵌入层
            self.Trans_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1)  # Transformer编码器
            self.sentence_embed =self.Transformer_embed  # 嵌入方法指向Transformer_embed
            return self.embed_size  # 返回输出维度
        if method == "gnn":
            embed_size = 128  # 嵌入维度
            self.edge_weight = nn.Embedding((self.vocab_size) * (self.vocab_size)+1, 1, padding_idx=0)  # 边权重嵌入
            self.node_embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=0)  # 节点嵌入
            self.node_weight = nn.Embedding(self.vocab_size, 1, padding_idx=0)  # 节点权重嵌入
            # nn.init.xavier_uniform_(self.edge_weight.weight)
            # nn.init.xavier_uniform_(self.node_weight.weight)
            self.sentence_embed = self.gnn_embed  # 嵌入方法指向gnn_embed
            return embed_size  # 返回输出维度
            # self.fc = nn.Sequential(
            #     nn.Linear(embed_size, 2),
            #     nn.ReLU(),
            #     nn.Dropout(self.dropout),
            #     nn.LogSoftmax(dim=1)
            # )


    def forward(self, node_subgraph, adj_subgraph, current_epoch=10, is_training=True):
        feat_subg = self.feat_full[node_subgraph]  # 取子图节点的特征
        label_subg = self.label_full[node_subgraph]  # 取子图节点的标签
        mask_subg = self.mask[node_subgraph]  # 取子图节点的mask
        feat_subg = self.sentence_embed(tokens=feat_subg, padding_mask=mask_subg, is_training=is_training)  # 句子嵌入
        # cover = stego = 0
        # for i in label_subg:
        #     if i[1] == 0:
        #         cover+=1
        #     else:
        #         stego+=1
        # print("cover: {}, stego: {}".format(cover, stego))
        try:
            label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]  # 多分类时转为类别索引
        except:
            print()
        if self.no_graph:
            pred_subg = self.classifier((None, feat_subg))[1]  # 只用句子嵌入做分类
        else:
            feat_subg_ = self.sentence_embed_norm(feat_subg)  # 归一化
            # sims = self.cos_sim(feat_subg_, is_training)
            # sims = self.top_sim(sims,topk=3).to(feat_subg_.device)
            # adj_subgraph += sims
            # adj_subgraph = adj_subgraph.to_dense()
            # # adj_subgraph = torch.ones_like(adj_subgraph).to(adj_subgraph.device) + adj_subgraph
            # adj_subgraph = adj_subgraph+sims*(sims>=0.85)
            # indices = adj_subgraph.to_sparse().indices()
            # values =  adj_subgraph.to_sparse().values()
            # adj_subgraph = torch.sparse_coo_tensor(indices,values,size=adj_subgraph.size())

            feat_subg_ = self.dropout_layer(feat_subg_)  # dropout
            # sims = self.sim(feat_subg.unsqueeze(1), feat_subg.unsqueeze(0))

            if current_epoch >= 0:
                _, emb_subg = self.conv_layers((adj_subgraph, feat_subg_))  # 图卷积
                emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)  # L2归一化
                if self.hidden_dim == -1:
                    # pred_subg = self.classifier((None, emb_subg_norm))[1]
                    pred_subg = self.classifier((None, torch.cat([emb_subg_norm, feat_subg],dim=1)))[1]  # 拼接特征做分类
                else:
                    pred_subg = self.classifier_((None, torch.cat([emb_subg_norm, feat_subg], dim=1)))[1]  # 先高阶聚合
                    pred_subg = self.classifier(pred_subg)  # 再线性分类
            else:
                pred_subg = self.classifier2((None, feat_subg))[1]  # 备用分类器
        return pred_subg, label_subg, label_subg_converted  # 返回预测、标签、转换标签


    def cnn_embed(self, tokens, padding_mask=None, is_training = None):
        '''
        CNN sentence embedding
        '''
        x = tokens.long()  # 转为long类型
        _ = self.embedding(x)  # 嵌入
        _ = _.permute(0, 2, 1)  # 调整维度
        result = []
        for self.cnn in self.cnn_list:
            __ = self.cnn(_)  # 卷积
            __ = self.max_pool(__)  # 池化
            __ = self.relu(__)  # 激活
            result.append(__.squeeze(dim=2))  # 去掉多余维度
        _ = torch.cat(result, dim=1)  # 拼接所有卷积输出
        return _  # 返回句子嵌入
        # return self.relu(self.avg_pool(self.embed(tokens.long()).permute(0,2,1)).squeeze())
        # return self.max_pool(self.sentence_encoder(self.embed(tokens.long()).permute(1,0,2), src_key_padding_mask=padding_mask).permute(1,0,2).permute(0,2,1)).squeeze()

    def pool_embed(self, tokens,padding_mask=None,is_training = None):
        return self.relu(self.embed_pool(self.embed(tokens.long()).permute(0, 2, 1)).squeeze())  # 池化嵌入

    def RNN_embed(self, tokens, padding_mask=None,is_training = None):
        x = tokens.long()  # 转为long类型
        _ = self.embed(x)  # 嵌入
        _ = _.permute(1, 0, 2)  # 调整维度
        # h_out = self.rnn(_)
        hidden = self.rnn(_)  # RNN前向
        hidden = hidden[1][-1].permute(1, 0, 2).reshape((-1, self.sentence_embedding_dim))  # 取最后一层隐藏状态
        return hidden  # 返回RNN嵌入

    def LSTM_embed(self, tokens, padding_mask=None,is_training = None):
        x = tokens.long()  # 转为long类型
        _ = self.embed(x)  # 嵌入
        _ = _.permute(1, 0, 2)  # 调整维度
        __, h_out = self.rnn(_)  # LSTM前向
        # if self._cell in ["lstm", "bi-lstm"]:
        #     h_out = torch.cat([h_out[0], h_out[1]], dim=2)
        h_out = torch.cat([h_out[0], h_out[1]], dim=2)  # 拼接正反向隐藏状态
        h_out = h_out.permute(1, 0, 2)  # 调整维度
        h_out = h_out.reshape(-1, h_out.shape[1] * h_out.shape[2])  # 展平
        return h_out  # 返回LSTM嵌入

    def LSTMATT_embed(self, tokens, padding_mask=None,is_training = None):
        input_lstm = self.embed(tokens.long())  # 嵌入
        input_lstm = input_lstm.permute(1,0,2)  # 调整维度
        output, _ = self.lstm(input_lstm)  # LSTM前向
        output = self.LSTMATT_DP(output)  # dropout
        # sentence_output = torch.cat([_[0], _[1]], dim=2)
        # scc, _ = self.attn(output, output, output, key_padding_mask=padding_mask)
        # scc, _ = self.attn(sentence_output.mean(dim=0).unsqueeze(0), output, output, key_padding_mask=padding_mask)
        # return scc.squeeze()
        return self.embed_pool(self.attn(output, output, output, key_padding_mask=padding_mask)[0].permute(1,0,2).permute(0,2,1)).squeeze()  # 注意力池化
        # return self.attn(output, output, output, key_padding_mask=padding_mask)[0].mean(dim=0)

    def Transformer_embed(self, tokens, padding_mask=None,is_training = None):
        return self.Trans_encoder(self.embed(tokens.long()).permute(1, 0, 2),src_key_padding_mask=padding_mask).mean(dim=0)  # Transformer编码后取均值

    def gnn_embed(self, tokens, padding_mask=None,is_training = None):
        X = tokens.long()  # 转为long类型
        NX, EW = self.get_neighbors(X, nb_neighbor=2)  # 获取邻居节点和边权重
        NX = NX.long()
        EW = EW.long()
        # NX = input_ids
        # EW = input_ids
        Ra = self.node_embedding(NX)  # 邻居节点嵌入
        # edge weight  (bz, seq_len, neighbor_num, 1)
        Ean = self.edge_weight(EW)  # 边权重嵌入
        # neighbor representation  (bz, seq_len, embed_dim)
        if not is_training:
            B, L, N, E = Ra.shape
            # Mn = torch.zeros(size=(B,L,E)).to(Ra.device)
            y = torch.zeros(size=(B,E)).to(Ra.device)  # 初始化输出
            Rn = self.node_embedding(X)  # 自身节点嵌入
            # self node weight  (bz, seq_len, 1)
            Nn = self.node_weight(X)  # 自身节点权重
            # aggregate node features
            for i in range(B):
                tmp = (Ra[i]*Ean[i]).max(dim=1)[0]  # 邻居特征加权后max pooling
                # Mn[i,:,:] = tmp
                y[i] = ((1-Nn[i])*tmp + Nn[i] * Rn[i]).sum(dim=0)  # 聚合邻居和自身特征
            return y  # 返回节点嵌入
        Mn = (Ra * Ean).max(dim=2)[0]  # max pooling
        # self representation (bz, seq_len, embed_dim)
        Rn = self.node_embedding(X)  # 自身节点嵌入
        # self node weight  (bz, seq_len, 1)
        Nn = self.node_weight(X)  # 自身节点权重
        # aggregate node features
        y = (1 - Nn) * Mn + Nn * Rn  # 聚合邻居和自身特征
        return y.sum(dim=1)  # 返回节点嵌入


    def get_neighbors(self, x_ids, nb_neighbor=2):
        B, L = x_ids.size()  # 批量和序列长度
        neighbours = torch.zeros(size=(L, B, 2 * nb_neighbor))  # 邻居节点初始化
        ew_ids = torch.zeros(size=(L, B, 2 * nb_neighbor))  # 边权重初始化
        # pad = [0] * nb_neighbor
        pad = torch.zeros(size=(B, nb_neighbor)).to(x_ids.device)  # padding
        # x_ids_ = pad + list(x_ids) + pad
        x_ids_ = torch.cat([pad, x_ids, pad], dim=-1)  # 拼接padding
        for i in range(nb_neighbor, L + nb_neighbor):
            # x = x_ids_[i - nb_neighbor: i] + x_ids_[i + 1: i + nb_neighbor + 1]
            neighbours[i - nb_neighbor, :, :] = torch.cat(
                [x_ids_[:, i - nb_neighbor: i], x_ids_[:, i + 1: i + nb_neighbor + 1]], dim=-1)  # 取左右邻居
        # ew_ids[i-nb_neighbor,:,:] = (x_ids[i-nb_neighbor,:] -1) * self.vocab_size + nb_neighbor[i-nb_neighbor,:,:]
        neighbours = neighbours.permute(1, 0, 2).to(x_ids.device)  # 调整维度
        ew_ids = ((x_ids) * (self.vocab_size)).reshape(B, L, 1) + neighbours  # 计算边权重id
        ew_ids[neighbours == 0] = 0  # padding位置权重为0
        return neighbours, ew_ids  # 返回邻居和边权重


    def _loss(self, preds, labels, norm_loss):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)
        """
        if args_global.train_rule == 'None':
            per_cls_weights = None 
        elif args_global.train_rule == 'Reweight':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, self.cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args_global.gpu)
        elif args_global.train_rule == 'DRW':
            idx = 0 if self.current_epoch < args_global.DRW_epoch else 1
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args_global.gpu)
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)  # 扩展维度
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss,reduction='sum')(preds, labels)  # 多标签二分类损失
        elif self.loss_type == "CE":
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)  # 多分类交叉熵损失
            return (norm_loss*_ls).sum()  # 加权求和
        elif self.loss_type == "LDAM":
            _ls = LDAMLoss(cls_num_list=self.cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args_global.gpu)
            return _ls(preds, labels)
        elif self.loss_type == 'Focal':
            _ls = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args_global.gpu)
            return _ls(preds, labels)
        elif self.loss_type == 'VSLoss':
            _ls = VSLoss(weight=per_cls_weights,cls_num_list=self.cls_num_list, tau=args_global.VSLoss_tau, gamma=args_global.VSLoss_gamma).cuda(args_global.gpu)
            return _ls(preds, labels)

    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        num_param = 0  # 参数计数
        aggregators = []  # 聚合器列表
        for l in range(self.num_layers):
            aggr = self.aggregator_cls(
                    *self.dims_weight[l],
                    dropout=self.dropout,
                    act=self.act_layer[l],
                    order=self.order_layer[l],
                    aggr=self.aggr_layer[l],
                    bias=self.bias_layer[l],
                    mulhead=self.mulhead,
            )  # 构造聚合器
            num_param += aggr.num_param  # 累加参数数
            aggregators.append(aggr)  # 添加到列表
        return aggregators, num_param  # 返回聚合器和参数数


    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)  # 根据损失类型返回概率

    #在这里修改了train_step函数，添加了SAM优化器
    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph,current_epoch=0):
        """
        Forward and backward propagation
        """
        if self.use_sam:
            self.optimizer = sam.SAM(self.parameters(), torch.optim.Adam, rho=self.rho, lr=0.01)
        self.train()  # 进入训练模式
        self.optimizer.zero_grad()  # 梯度清零
        preds, labels, labels_converted = self(node_subgraph, adj_subgraph,current_epoch=current_epoch, is_training=True)  # 前向传播
        loss = self._loss(preds, labels_converted, norm_loss_subgraph) # 计算损失
        return_loss = loss
        return_preds = preds
        return_labels = labels
        if self.use_sam:
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            preds, labels, labels_converted = self(node_subgraph, adj_subgraph,current_epoch=current_epoch, is_training=True)  # 前向传播
            loss = self._loss(preds, labels_converted, norm_loss_subgraph) # 计算损失
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 5)  # 梯度裁剪
            self.optimizer.second_step(zero_grad=False)
        else:
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm(self.parameters(), 5)  # 梯度裁剪
            self.optimizer.step()  # 优化器更新
        return return_loss, self.predict(return_preds), return_labels  # 返回损失、预测、标签


    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Forward propagation only
        """
        self.eval()  # 进入评估模式
        with torch.no_grad():  # 关闭梯度
            preds,labels,labels_converted = self(node_subgraph, adj_subgraph,is_training=False)  # 前向传播
            loss = self._loss(preds,labels_converted,norm_loss_subgraph)  # 计算损失
        return loss, self.predict(preds), labels  # 返回损失、预测、标签
