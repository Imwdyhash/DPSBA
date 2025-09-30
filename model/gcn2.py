import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from utils.graph import numpy_to_graph

gcn_msg = fn.copy_u('h', 'm')
gcn_reduce = fn.sum(msg='m', out='h')

import torch
import torch.nn as nn
import torch.nn.functional as F

#重新实现的可以实现梯度更新的版本
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, A, X):
        """
        A: 邻接矩阵 (B, N, N)
        X: 节点特征 (B, N, in_feats)
        """

        support = torch.bmm(A, X)  # (B, N, in_feats)
        out = self.linear(support)  # (B, N, out_feats)

        # 调整维度以适应 BatchNorm1d
        B, N, C = out.shape
        out = out.view(B * N, C)  # (B * N, out_feats)
        out = self.bn(out)
        out = out.view(B, N, C)  # (B, N, out_feats)

        # 应用 ReLU 激活函数

        return out  # 经过 ReLU 激活


# 2层 GCN
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_dim, hidden_dim[0]))
        self.layers.append(nn.Dropout(p=dropout))

        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNLayer(hidden_dim[i], hidden_dim[i + 1]))

        fc = []

        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        """
        data[0]: 节点特征 (B, N, F)
        data[1]: 邻接矩阵 (B, N, N)
        data[2]: 节点 mask (B, N) - 0 或 1，指示哪些节点有效
        """
        #if isinstance(data[1], torch.Tensor) and len(data.shape) == 2 and data.shape[0] == 2:


        A = data[1]  # 邻接矩阵 (B, N, N)
        X = data[0]  # 节点特征 (B, N, F)
        mask = data[2]  # 节点 mask (B, N)
        # print("A,X requires_grad:", A.requires_grad,X.requires_grad)

        #print("A.requires_grad:", A.requires_grad, "X.requires_grad:", X.requires_grad)

        # 处理 mask
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)  # (B, N, 1)

        B, N, FF = X.shape
        X = X.reshape(B, N, FF)  # 维度 (B, N, F)
        mask = mask.reshape(B, N, 1)

        X = F.relu(self.layers[0](A, X))
        X = X * mask
        # 通过 GCN 层
        for layer in self.layers[2:]:
            X = layer(A, X)
            X = X * mask

        F_prime = X.shape[-1]
        X = X.reshape(B, N, F_prime)
        X = torch.max(X, dim=1)[0].squeeze()  # (B, F_prime)

        # 全连接层
        X = self.fc(X)

        return X

