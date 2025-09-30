import torch
import torch.nn as nn
import torch.nn.functional as F


# 实现 GraphSAGE 聚合层
class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SAGEConv, self).__init__()
        self.linear_self = nn.Linear(in_feats, out_feats)
        self.linear_neigh = nn.Linear(in_feats, out_feats)

    def forward(self, A, X):
        """
        A: 邻接矩阵 (B, N, N)
        X: 节点特征 (B, N, in_feats)
        """
        # 聚合邻居特征
        neigh_feats = torch.bmm(A, X)  # (B, N, in_feats)

        # 计算自身特征和邻居特征的线性变换
        self_feats = self.linear_self(X)
        neigh_feats = self.linear_neigh(neigh_feats)

        # 合并自身特征和邻居特征
        combined = self_feats + neigh_feats

        # 归一化
        combined = F.normalize(combined, p=2, dim=-1)

        return F.relu(combined)


# 实现 SAGPool 层
class SAGPool(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(SAGPool, self).__init__()
        self.ratio = ratio
        self.attn = nn.Linear(in_channels, 1)

    def forward(self, A, X, mask=None):
        """
        A: 邻接矩阵 (B, N, N)
        X: 节点特征 (B, N, in_channels)
        mask: 节点 mask (B, N) - 0 或 1，指示哪些节点有效
        """
        # 计算注意力分数
        scores = self.attn(X).squeeze(-1)  # (B, N)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算每个图的选择数量
        num_nodes = X.size(1)
        num_selected = int(self.ratio * num_nodes)

        # 选择前 k 个节点
        _, topk_indices = torch.topk(scores, num_selected, dim=1)  # (B, k)

        # 提取选择的节点特征
        batch_size = X.size(0)
        selected_X = []
        selected_A = []
        for i in range(batch_size):
            idx = topk_indices[i]
            selected_X.append(X[i, idx])
            selected_A.append(A[i, idx][:, idx])

        selected_X = torch.stack(selected_X, dim=0)  # (B, k, in_channels)
        selected_A = torch.stack(selected_A, dim=0)  # (B, k, k)

        return selected_A, selected_X


# 包含 SAGPool 的 GraphSAGE 模型
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.2, pool_ratio=0.5):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        self.layers.append(SAGEConv(in_dim, hidden_dim[0]))
        self.pool_layers.append(SAGPool(hidden_dim[0], ratio=pool_ratio))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(SAGEConv(hidden_dim[i], hidden_dim[i + 1]))
            self.pool_layers.append(SAGPool(hidden_dim[i + 1], ratio=pool_ratio))

        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
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

        B, N, F = X.shape
        X = X.reshape(B, N, F)  # 维度 (B, N, F)
        mask = mask.reshape(B, N, 1)

        # 通过 GraphSAGE 层和 SAGPool 层
        for layer, pool_layer in zip(self.layers, self.pool_layers):
            X = layer(A, X)
            if mask is not None:
                X = X * mask
            if mask is not None:
                mask_to_use = mask.squeeze(-1)
            else:
                mask_to_use = None
            A, X = pool_layer(A, X, mask_to_use)
            mask = None  # 后续不再使用 mask

        F_prime = X.shape[-1]
        X = X.reshape(B, -1, F_prime)
        X = torch.max(X, dim=1)[0].squeeze()  # (B, F_prime)

        # 全连接层
        X = self.fc(X)

        return X
