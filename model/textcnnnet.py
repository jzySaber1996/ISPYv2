import torch
import torch.nn as nn
import torch.nn.functional as F


class textCNN(nn.Module):
    def __init__(self, embed_dim, kernel_num, kernel_sizes, dropout, Cla, pooling_size):
        super(textCNN, self).__init__()
        Dim = embed_dim  ##每个词向量长度
        self.Ci = 1  ##输入的channel数
        self.Knum = kernel_num  ## 每种卷积核的数量
        self.Ks = kernel_sizes  ## 卷积核list，形如[2,3,4]
        self.convs0 = nn.ModuleList([nn.Conv2d(self.Ci, self.Knum[0], (1, K)) for K in self.Ks])
        self.convs1 = nn.ModuleList([nn.Conv2d(self.Ci, self.Knum[1], (1, K)) for K in self.Ks])
        self.convs2 = nn.ModuleList([nn.Conv2d(self.Ci, self.Knum[2], (1, K)) for K in self.Ks])

        # self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (1, K)) for K in Ks])  ## 卷积层
        # self.pooling = nn.MaxPool1d(pooling_size)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        convs_list = [self.convs0, self.convs1, self.convs2]
        for convs, k_num in zip(convs_list, self.Knum):
            x = x.unsqueeze(1)  # (N,Ci,W,D)
            x = [F.elu(conv(x)) for conv in convs]  # len(Ks)*(N,Knum,W)
            x = [F.max_pool2d(line, (1, line.size(3))).squeeze(3).permute(0, 2, 1) for line in x]  # len(Ks)*(N,Knum)
            # x = [self.pooling(line).squeeze(3) for line in x]

            x = torch.cat(x, dim=2)  # (N,Knum*len(Ks))
            x = self.dropout(x)

        # logit = self.fc(x)
        return x
