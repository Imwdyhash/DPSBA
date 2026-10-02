# -*- coding: utf-8 -*-
"""
DPSBA AUC 复现桥接脚本 v2（修正训练样本太少的问题）
- 训练集：从原始 FRANKENSTEIN 读取全部干净训练图（约 2000 张）
- 测试集：后门图（DPSBA 生成）+ 等量干净测试图
- 用 SIGNET 训练异常检测器，计算 ROC-AUC
"""
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, r'D:\SIGNET')
sys.path.insert(0, r'D:\DPSBA')
from models import GIN, Explainer_GIN, HyperGNN, Explainer_MLP
from utils.datareader import DataReader

# ---------- SIGNET 模型（同 v1） ----------
class SIGNET(nn.Module):
    def __init__(self, input_dim, input_dim_edge, args, device):
        super(SIGNET, self).__init__()
        self.device = device
        self.embedding_dim = args['hidden_dim']
        if args['readout'] == 'concat':
            self.embedding_dim *= args['encoder_layers']
        if args['explainer_model'] == 'mlp':
            self.explainer = Explainer_MLP(input_dim, args['explainer_hidden_dim'], args['explainer_layers'])
        else:
            self.explainer = Explainer_GIN(input_dim, args['explainer_hidden_dim'],
                                           args['explainer_layers'], args['explainer_readout'])
        self.encoder = GIN(input_dim, args['hidden_dim'], args['encoder_layers'], args['pooling'], args['readout'])
        self.encoder_hyper = HyperGNN(input_dim, input_dim_edge, args['hidden_dim'],
                                      args['encoder_layers'], args['pooling'], args['readout'])
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_hyper = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                             nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        node_imp = self.explainer(data.x, data.edge_index, data.batch)
        edge_imp = self.lift_node_score_to_edge_score(node_imp, data.edge_index)
        y, _ = self.encoder(data.x, data.edge_index, data.batch, node_imp)
        y_hyper, _ = self.encoder_hyper(data.x, data.edge_index, data.edge_attr, data.batch, edge_imp)
        y = self.proj_head(y)
        y_hyper = self.proj_head_hyper(y_hyper)
        return y, y_hyper, node_imp, edge_imp

    @staticmethod
    def loss_nce(x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)
        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        return (loss_0 + loss_1) / 2.0

    def lift_node_score_to_edge_score(self, node_score, edge_index):
        src_lifted_att = node_score[edge_index[0]]
        dst_lifted_att = node_score[edge_index[1]]
        return src_lifted_att * dst_lifted_att

def adj_to_data(adj, feat, label):
    adj = np.array(adj)
    feat = np.array(feat, dtype=np.float32)
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    x = torch.tensor(feat, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=None, y=torch.tensor([label], dtype=torch.long))

def main():
    bkd_path = sys.argv[1] if len(sys.argv) > 1 else None
    if bkd_path is None:
        import glob
        candidates = glob.glob(r'D:\DPSBA\save\model\data\FRANKENSTEIN*')
        bkd_path = max(candidates, key=os.path.getmtime)
    print('读取后门数据:', bkd_path)
    d = torch.load(bkd_path, map_location='cpu')
    bkd_adj, bkd_feat = d['bkd_adj'], d['bkd_feat']
    bkd_gids = d['bkd_gids']
    n_bkd = len(bkd_adj)
    print(f'后门图 {n_bkd} 张')

    # ---- 从原始数据读全部图（干净图来源） ----
    class Args:
        pass
    args = Args()
    args.data_path = r'D:\DPSBA\dataset'
    args.dataset = 'FRANKENSTEIN'
    args.use_nlabel_asfeat = False
    args.use_org_node_attr = True
    args.use_degree_asfeat = False
    args.seed = 123
    args.train_ratio = 0.5
    args.data_verbose = False
    dr = DataReader(args)
    all_adj = dr.data['adj_list']
    all_feat = dr.data['features']
    train_gids = list(dr.data['splits']['train'])
    test_gids = list(dr.data['splits']['test'])

    # 干净训练图 = 训练集 - 后门图（约 2000 张，喂给 SIGNET 训练）
    bkd_set = set(bkd_gids)
    clean_train_gids = [g for g in train_gids if g not in bkd_set]
    # 干净测试图：从测试集里取和后门图等量
    clean_test_gids = test_gids[:n_bkd]
    print(f'SIGNET 训练干净图 {len(clean_train_gids)} 张，测试干净图 {len(clean_test_gids)} 张')

    n_feat = all_feat[0].shape[1]
    train_data = [adj_to_data(all_adj[g], all_feat[g], 0) for g in clean_train_gids]
    test_clean = [adj_to_data(all_adj[g], all_feat[g], 0) for g in clean_test_gids]
    test_bkd = [adj_to_data(a, f, 1) for a, f in zip(bkd_adj, bkd_feat)]
    test_data = test_clean + test_bkd
    print(f'测试集共 {len(test_data)} 张（干净 {len(test_clean)} + 后门 {len(test_bkd)}）')

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=9999, shuffle=False)

    args_sign = {
        'hidden_dim': 16, 'encoder_layers': 5, 'pooling': 'add', 'readout': 'concat',
        'explainer_model': 'gin', 'explainer_layers': 5, 'explainer_hidden_dim': 8,
        'explainer_readout': 'add',
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('设备:', device)

    epochs = 500  # 对齐 DPSBA config.py 的 SIGNET epochs 默认值
    n_trials = 5  # 跑 5 个随机种子，取均值±标准差（与论文 5 个 seed 口径完全一致）

    trial_aucs = []
    for trial in range(n_trials):
        # 固定随机种子（和 SIGNET 官方 main.py 的 set_seed 一致）
        import random as _random
        _random.seed(trial)
        np.random.seed(trial)
        torch.manual_seed(trial)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(trial)
            torch.cuda.manual_seed_all(trial)

        model = SIGNET(n_feat, 0, args_sign, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        best_auc = 0.0
        final_auc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            loss_all, num_sample = 0, 0
            for data in train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                y, y_hyper, _, _ = model(data)
                loss = model.loss_nce(y, y_hyper).mean()
                loss_all += loss.item() * data.num_graphs
                num_sample += data.num_graphs
                loss.backward()
                optimizer.step()

            model.eval()
            all_true, all_score = [], []
            with torch.no_grad():
                for data in test_loader:
                    all_true.append(data.y.cpu())
                    data = data.to(device)
                    y, y_hyper, _, _ = model(data)
                    all_score.append(model.loss_nce(y, y_hyper).cpu())
            ad_true = torch.cat(all_true).numpy()
            ad_score = torch.cat(all_score).numpy()
            ad_auc = roc_auc_score(ad_true, ad_score)
            final_auc = ad_auc
            best_auc = max(best_auc, ad_auc)
            if epoch % 100 == 0 or epoch == epochs:
                print(f'[seed {trial}] Epoch {epoch:3d} | Loss {loss_all/num_sample:.4f} | AUC {ad_auc*100:.2f}%')

        trial_aucs.append(final_auc * 100)
        print(f'[seed {trial}] 完成，最终 AUC: {final_auc*100:.2f}%（该 seed 最好 {best_auc*100:.2f}%）')

    mean_auc = np.mean(trial_aucs)
    std_auc = np.std(trial_aucs)
    print('=' * 50)
    print(f'5 个 seed 的 AUC: {[f"{a:.2f}" for a in trial_aucs]}')
    print(f'均值±标准差: {mean_auc:.2f}% ± {std_auc:.2f}%')
    print('论文表1 FRANKENSTEIN+GCN 的 AUC: 68.96%（5 个 seed 均值）')

if __name__ == '__main__':
    main()
