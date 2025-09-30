import torch
import torch.nn as nn
import torch.nn.functional as F

# GIN 层
class GINLayer(nn.Module):
    def __init__(self, in_feats, out_feats, eps=0):
        super(GINLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.eps = nn.Parameter(torch.FloatTensor([eps]))

    def forward(self, A, X):
        """
        A: 邻接矩阵 (B, N, N)
        X: 节点特征 (B, N, in_feats)
        """
        # 聚合邻居特征
        neighbor_agg = torch.bmm(A, X)  # (B, N, in_feats)
        # 加上自身特征
        self_feat = (1 + self.eps) * X
        combined_feat = neighbor_agg + self_feat  # (B, N, in_feats)

        out = self.linear(combined_feat)  # (B, N, out_feats)

        # 调整维度以适应 BatchNorm1d
        B, N, C = out.shape
        out = out.view(B * N, C)  # (B * N, out_feats)
        out = self.bn(out)
        out = out.view(B, N, C)  # (B, N, out_feats)

        # 应用 ReLU 激活函数
        out = F.relu(out)

        return out

# 2 层 GIN
class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.2, eps=0):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GINLayer(in_dim, hidden_dim[0], eps))
        self.layers.append(nn.Dropout(p=dropout))

        for i in range(len(hidden_dim) - 1):
            self.layers.append(GINLayer(hidden_dim[i], hidden_dim[i + 1], eps))

        fc = []
        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        """
        data[0]: 节点特征 (B, N, F)
        data[1]: 邻接矩阵 (B, N, N)
        data[2]: 节点 mask (B, N) - 0 或 1，指示哪些节点有效
        """
        A = data[1]  # 邻接矩阵 (B, N, N)
        X = data[0]  # 节点特征 (B, N, F)
        mask = data[2]  # 节点 mask (B, N)

        # 处理 mask
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)  # (B, N, 1)

        B, N, FF = X.shape
        X = X.reshape(B, N, FF)  # 维度 (B, N, F)
        mask = mask.reshape(B, N, 1)

        X = self.layers[0](A, X)
        X = X * mask
        # 通过 GIN 层
        for layer in self.layers[2:]:
            X = layer(A, X)
            X = X * mask

        F_prime = X.shape[-1]
        X = X.reshape(B, N, F_prime)
        X = torch.max(X, dim=1)[0].squeeze()  # (B, F_prime)

        # 全连接层
        X = self.fc(X)

        return X
