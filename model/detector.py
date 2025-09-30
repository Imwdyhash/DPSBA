import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch_scatter import scatter_mean,scatter_max
# 定义 GCN 模型
class Detector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Detector, self).__init__()
        # 第一个 GCN 层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 第二个 GCN 层
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self,data):
        x =data.x
        edge_index=data.edge_index
        # 第一个 GCN 层前向传播并使用 ReLU 激活函数
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 随机失活防止过拟合
        x = F.dropout(x, p=0.2, training=self.training)
        # 第二个 GCN 层前向传播
        x = self.conv2(x, edge_index)
        #x=torch.sigmoid(x)
        x= scatter_mean(x, data.batch, dim=0)
        x = F.log_softmax(x, dim=1)

        return x



class MLPDetector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLPDetector, self).__init__()
        # 第一个全连接层
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        # 第二个全连接层
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # 获取图的节点特征
        x = data.x
        # 第一个全连接层前向传播并使用 ReLU 激活函数
        x = self.fc1(x)
        x = F.relu(x)
        # 随机失活防止过拟合
        x = F.dropout(x, p=0.2, training=self.training)
        # 第二个全连接层前向传播
        x = self.fc2(x)
        # 按图进行平均池化
        #x = scatter_mean(x, data.batch, dim=0)
        #最大池化
        x, _ = scatter_max(x, data.batch, dim=0)
        # 使用 log_softmax 函数进行二分类
        x = F.log_softmax(x, dim=1)
        return x

