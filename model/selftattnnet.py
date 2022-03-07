import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import os
import numpy as np
import logging
from tqdm import tqdm

class SelfAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, batch_size, concat=True):
        super(SelfAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活
        self.batch_size = batch_size

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, weight_tensor, d_window):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        """
        h = torch.matmul(inp, self.W)  # [N, out_features]
        N = h.size()[1]  # N 图的节点数
        N_BATCH = h.size()[0]

        a_input = torch.cat([h.repeat(1, 1, N).view(N_BATCH, N * N, -1), h.repeat(1, N, 1)], dim=2) \
            .view(N_BATCH, N, -1, 2 * self.out_features)
        # [batchsize, N, N, 2*out_features]
        e_attn = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        e_attn = torch.mul(e_attn, weight_tensor)
        # [batchsize, N, N, 1] => [batchsize, N, N] 图注意力的相关系数（未归一化）
        zero_vec = -1e12 * torch.ones_like(e_attn)  # 将没有连接的边置为负无穷
        adj_mat = torch.tensor([[1 if np.abs(i - j) <= d_window else 0 for i in range(e_attn.size()[1])]
                   for j in range(e_attn.size()[1])]).repeat(e_attn.size()[0], 1, 1).cuda() # Set the local attention matrix
        attention = torch.where(adj_mat > 0, e_attn, zero_vec)
        # attention = torch.where(adj > 0, e, zero_vec)  # [batchsize, N, N]

        attention = F.softmax(attention, dim=2)  # softmax形状保持不变 [batchsize, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        attention = torch.tensor(attention, dtype=torch.float32).cuda()
        h_prime = torch.matmul(attention, h)  # [batchsize, N, N].[batchsize, N, out_features] => [batchsize, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads, batch_size):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(SAT, self).__init__()
        self.dropout = dropout
        self.batch_size = batch_size

        # 定义multi-head的图注意力层，List不是Module类型要依次加入
        self.attentions = [SelfAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, batch_size=batch_size, concat=True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = SelfAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, batch_size=batch_size, concat=False)

    def forward(self, x, weight_tensor, d_window):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, weight_tensor, d_window) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, weight_tensor, d_window))  # 输出并激活
        # return F.log_softmax(x, dim=2)  # log_softmax速度变快，保持数值稳定
        return x  # log_softmax速度变快，保持数值稳定


if __name__ == "__main__":
    gat = SAT(10, 10, 3, 0.2, 0.2, 4, 2)
    x = torch.randn(2, 5, 10)
    print(gat(x))
