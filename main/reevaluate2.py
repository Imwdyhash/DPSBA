import sys, os
sys.path.append(os.path.abspath('../..'))

import time
import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from utils.datareader import GraphData, DataReader
from utils.batch import collate_batch
from model.gcn2 import GCN
from model.gat import GAT
from model.sage import GraphSAGE
from model.gin2 import GIN
from config import parse_args
from trojan.prop import evaluate,train_model,train_model2
import os
import ast

cuda = torch.device('cuda')


def run(args,filter_id=None):



    if args.retest:
        filter_id = set([])
        filter_id_test = []
    bkd_path = os.path.join(args.bkd_data_save_path,args.dataset)#bkd_model_save_path

    loaded_data = torch.load(bkd_path)
    benign_dr = loaded_data['benign_dr']
    bkd_gids_train=loaded_data['bkd_gids_train']#实际的后门id
    bkd_dr_test = loaded_data['bkd_dr_test']
    bkd_gids_test = loaded_data['bkd_gids_test']

    common_elements = torch.isin(torch.tensor(list(filter_id)), torch.tensor(bkd_gids_train))
    common_elements2 = torch.isin(torch.tensor(list(filter_id_test)), torch.tensor(bkd_gids_test))
    count = torch.sum(common_elements).item()
    count2 = torch.sum(common_elements2).item()
    print('训练集',len(filter_id),len(bkd_gids_train),'命中',count)
    print('测试集',len(filter_id_test), len(bkd_gids_test), '命中', count2)

    bkd_dr_train = loaded_data['bkd_dr_train']
    # for gid in bkd_gids_train:
    #     print(bkd_dr_train.data['adj_list'][gid]==benign_dr.data['adj_list'][gid],bkd_dr_train.data['adj_list'][gid])

    #把筛选出来的可疑后门图复原
    # for gid in list(filter_id):#bkd_gids_train filter_id 假设全部恢复也就是没有后门训练集
    #     bkd_dr_train.data['adj_list'][gid]=benign_dr.data['adj_list'][gid]
    #     bkd_dr_train.data['features'][gid] = benign_dr.data['features'][gid]
    #     bkd_dr_train.data['labels'][gid] = benign_dr.data['labels'][gid]#针对非clean_label的情况



    original_train_gids = set(bkd_dr_train.data['splits']['train'])#原来的训练集id
    assert filter_id.issubset(original_train_gids), "filter mistake"
    train_gids = list(original_train_gids - filter_id)  # 筛选后的训练集id
    test_gids = bkd_dr_train.data['splits']['test']
    yt_gids = [gid for gid in bkd_gids_test
               if benign_dr.data['labels'][gid] == args.target_class]
    yx_gids = list(set(bkd_gids_test) - set(yt_gids))
    clean_graphs_test = list(set(benign_dr.data['splits']['test']) - set(bkd_gids_test))

    gdata_train = GraphData(bkd_dr_train, train_gids)#筛选后的选练级
    loader_train = DataLoader(gdata_train,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=collate_batch)
    loaders = {}
    loaders['train'] = loader_train

    gdata_test = GraphData(bkd_dr_test, yx_gids)
    loader_test = DataLoader(gdata_test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_batch)
    gdata_test2 = GraphData(benign_dr, bkd_gids_test)
    loader_test2 = DataLoader(gdata_test2,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_batch)
    loaders['test'] = loader_test
    loaders['test2'] = loader_test2
    print('train %d, test %d' % (len(loaders['train'].dataset), len(loaders['test'].dataset)))


    # prepare model
    in_dim = loaders['train'].dataset.num_features
    out_dim = loaders['train'].dataset.num_classes
    if args.model == 'gcn':
        model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'gat':
        model = GAT(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, num_heads=args.num_head)
    elif args.model == 'sage':
        model = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'gin':
        model = GIN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise NotImplementedError(args.model)



    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # training
    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)

    model.to(cuda)


    pset = bkd_gids_train # 正样本就是训练集中被选中插入trigger的图 也就是要被异常检测找出来的图
    nset = list(set(original_train_gids) - set(pset))#干净的部分

    train_model2(args, bkd_dr_train, model, list(set(pset)), list(set(nset)),bkd_dr_test,bkd_gids_test,yx_gids,clean_graphs_test)
    cpu = torch.device('cpu')
    model.to(cpu)





    bkd_acc = evaluate(args, bkd_dr_test, model, bkd_gids_test)  # 所有的中毒样本
    flip_rate = evaluate(args, bkd_dr_test, model, yx_gids)  # 要被攻击翻转的样本（也就是原来非target的中毒样本）
    clean_acc = evaluate(args, bkd_dr_test, model, clean_graphs_test)  # 测试集里的干净图（用来测CAD的）
    print('asr', bkd_acc, 'flip_rate', flip_rate, 'clean_acc', clean_acc)

if __name__ == '__main__':
    args = parse_args()
    run(args)