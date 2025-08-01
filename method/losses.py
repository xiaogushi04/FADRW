import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------
# 实现简易函数版本的损失 (用于复用或单独调用)
# -----------------------------

def focal_loss(input_values, gamma):
    """Computes the focal loss given input loss values and focusing parameter gamma"""
    # input_values: 通常是交叉熵损失值
    # p = exp(-input) 得到概率估计
    p = torch.exp(-input_values)
    # focal loss: (1 - p)^gamma * loss
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


def ib_loss(input_values, ib):
    """Computes influence-balanced loss: 按 ib 权重对损失加权"""
    # input_values: per-sample loss, ib: per-sample权重
    loss = input_values * ib
    return loss.mean()


# -----------------------------
# IBLoss：Influence-Balanced Loss
# -----------------------------
class IBLoss(nn.Module):
    def __init__(self, num_classes, weight=None, alpha=10000.):
        """
        num_classes: 分类类别数
        weight: 类别权重，可用于不平衡加权
        alpha: 用于平衡梯度与特征的常数，越大对尾部重视越高
        """
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 1e-3  # 防止除零
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, input, target, features):
        """
        input: 模型输出 logits (N, C)
        target: 真值标签 (N,)
        features: 每个样本对应的特征强度（如中间层norm值），(N,)
        """
        # 计算 per-sample 的梯度强度指标 grads: ||softmax - one_hot||_1
        grads = torch.sum(
            torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)),
            dim=1
        )  # (N,)
        # ib 权重 = alpha / (grads * features + epsilon)
        ib = self.alpha / (grads * features.reshape(-1) + self.epsilon)
        # 使用带权重的基础交叉熵损失
        base_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        return ib_loss(base_loss, ib)


# -----------------------------
# FocalLoss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        """
        weight: 类别不平衡权重
        gamma: focusing 参数，越大对难样本关注越强
        """
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        # 计算 per-sample 交叉熵损失，再调用 focal_loss
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        return focal_loss(ce_loss, self.gamma)


# -----------------------------
# LDAMLoss：Label-Distribution-Aware Margin Loss
# -----------------------------
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        cls_num_list: 各类别样本数列表
        max_m: margin 最大值
        s: logit scale 因子
        """
        super(LDAMLoss, self).__init__()
        # 根据类别频率反算 margin 列表
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.cuda.FloatTensor(m_list)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        # 生成 one-hot mask
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        # 每个样本对应的 margin
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        # 调整 logits
        x_m = x - batch_m
        # 对目标类别用 margin 后的 logit，其他类别保持原始
        output = torch.where(index, x_m, x)
        # 放大并计算交叉熵
        return F.cross_entropy(self.s * output, target, weight=self.weight)


# -----------------------------
# VSLoss：Vector Scaling Loss
# -----------------------------
class VSLoss(nn.Module):
    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        """
        cls_num_list: 各类别样本数列表
        gamma: 调节类别采样概率幂
        tau: logit 偏移因子
        """
        super(VSLoss, self).__init__()
        # 计算类别概率分布
        cls_probs = [n / sum(cls_num_list) for n in cls_num_list]
        # 计算缩放因子 Delta
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)
        # 计算偏移量 iota
        iota_list = tau * np.log(cls_probs)
        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(temp)
        self.weight = weight

    def forward(self, x, target):
        # 对 logit 做缩放和平移
        output = x / self.Delta_list + self.iota_list
        return F.cross_entropy(output, target, weight=self.weight)
