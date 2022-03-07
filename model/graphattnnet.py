import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import os
import numpy as np
import logging
from tqdm import tqdm


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, batch_size, concat=True):
        super(GraphAttentionLayer, self).__init__()
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

    def forward(self, inp, adj_weights):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.matmul(inp, self.W)  # [N, out_features]
        N = h.size()[1]  # N 图的节点数
        N_BATCH = h.size()[0]

        a_input = torch.cat([h.repeat(1, 1, N).view(N_BATCH, N * N, -1), h.repeat(1, N, 1)], dim=2) \
            .view(N_BATCH, N, -1, 2 * self.out_features)
        # [batchsize, N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [batchsize, N, N, 1] => [batchsize, N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        weighted_e = torch.mul(adj_weights, e)
        attention = torch.where(adj_weights > 0, weighted_e, zero_vec)  # [batchsize, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=2)  # softmax形状保持不变 [batchsize, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention,
                               h)  # [batchsize, N, N].[batchsize, N, out_features] => [batchsize, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads, batch_size):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.batch_size = batch_size

        # 定义multi-head的图注意力层，List不是Module类型要依次加入
        self.attentions = [
            GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, batch_size=batch_size, concat=True)
            for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha,
                                           batch_size=batch_size, concat=False)

    def forward(self, x, adj, d_window):
        weight_attn = self.init_weight_graph(adj, d_window)
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, weight_attn) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, weight_attn))  # 输出并激活
        # return F.log_softmax(x, dim=2)  # log_softmax速度变快，保持数值稳定
        return x  # log_softmax速度变快，保持数值稳定

    def init_weight_graph(self, adj, d_window):
        n_batch = adj.size()[0]
        weight_batch = 0
        for t_batch in range(n_batch):
            adj_raw = torch.tensor(adj[t_batch], dtype=torch.float).cuda()
            adj_batch_raw = adj_raw.repeat(n_batch, 1, 1)
            adj_graphs = [adj_batch_raw]
            previous_weight_matrix = adj_raw
            weight_attn_matrix = torch.eye(adj_raw.size()[0]).cuda()
            init_weight_attn = previous_weight_matrix - weight_attn_matrix
            weight_attn_matrix += (np.exp(-(1 ** 2) / (2 * (d_window / 2) ** 2))) * init_weight_attn
            for i in range(d_window - 1):
                e_previous_weights = torch.matmul(previous_weight_matrix, adj_raw.permute(1, 0))
                one_vec = torch.ones_like(e_previous_weights).cuda()  # Set 1 matrix
                zero_vec = torch.zeros_like(e_previous_weights).cuda()
                e_previous_weights = torch.tensor(e_previous_weights, dtype=torch.float).cuda()
                e_adj_ret = torch.where(e_previous_weights > 0, one_vec, zero_vec)
                e_adj = e_adj_ret - previous_weight_matrix
                e_adj_graph = torch.where(e_adj > 0, one_vec, zero_vec)
                e_adj_batch_graph = e_adj_graph.repeat(n_batch, 1, 1)
                weight_attn_matrix += (np.exp(-((i + 2) ** 2) / (2 * (d_window / 2) ** 2))) * e_adj_graph
                previous_weight_matrix = e_adj_ret
                adj_graphs.append(e_adj_batch_graph)
            if t_batch == 0:
                weight_batch = weight_attn_matrix.unsqueeze(0)
            else:
                weight_batch = torch.cat((weight_batch, weight_attn_matrix.unsqueeze(0)), dim=0)
        return weight_batch


if __name__ == "__main__":
    gat = GAT(10, 10, 3, 0.2, 0.2, 4, 2)
    x = torch.randn(2, 5, 10)
    adj = torch.tensor([[[1, 1, 0, 1, 1],
                         [1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [1, 0, 1, 1, 1],
                         [1, 0, 0, 1, 1]],
                        [[1, 1, 0, 1, 1],
                         [1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [1, 0, 1, 1, 1],
                         [1, 0, 0, 1, 1]]])
    print(gat(x, adj, 3))
