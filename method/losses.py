import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss given input loss values and focusing parameter gamma"""
    p = torch.exp(-input_values)
    # focal loss: (1 - p)^gamma * loss
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


# -----------------------------
# FocalLoss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        return focal_loss(ce_loss, self.gamma)


# -----------------------------
# LDAMLoss：Label-Distribution-Aware Margin Loss
# -----------------------------
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.cuda.FloatTensor(m_list)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


# -----------------------------
# FADRW：Vector Scaling Loss      
# -----------------------------
class FADRW(nn.Module):
    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()
        cls_probs = [n / sum(cls_num_list) for n in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)
        iota_list = tau * np.log(cls_probs)
        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(temp)
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list
        return F.cross_entropy(output, target, weight=self.weight)
