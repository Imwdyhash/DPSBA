import os
import sys
from torch_geometric.data import Data
from utils.datareader import DataReader
sys.path.append('/home/zxx5113/BackdoorGNN/')
import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import dense_to_sparse
def select_cdd_graphs_random(args, data: list, adj_list: list, subset: str):#不区分是否clean_label混合着选
    '''
    Given a data (train/test), (randomly or determinately)
    pick up some graph to put backdoor information, return ids.
    '''
    rs = np.random.RandomState(args.seed)
    graph_sizes = [np.array(adj).shape[0] for adj in adj_list]
    bkd_graph_ratio = args.bkd_gratio_train if subset == 'train' else args.bkd_gratio_test
    bkd_num = int(np.ceil(bkd_graph_ratio * len(data)))

    assert len(data) > bkd_num, "Graph Instances are not enough"
    picked_ids = []

    # Randomly pick up graphs as backdoor candidates from data
    remained_set = copy.deepcopy(data)
    loopcount = 0
    while bkd_num - len(picked_ids) > 0 and len(remained_set) > 0 and loopcount <= 50:
        loopcount += 1

        cdd_ids = rs.choice(remained_set, bkd_num - len(picked_ids), replace=False)
        for gid in cdd_ids:
            if bkd_num - len(picked_ids) <= 0:
                break
            gsize = graph_sizes[gid]
            if gsize >= 3 * args.bkd_size * args.bkd_num_pergraph:
                picked_ids.append(gid)

        if len(remained_set) < len(data):
            for gid in cdd_ids:
                if bkd_num - len(picked_ids) <= 0:
                    break
                gsize = graph_sizes[gid]
                if gsize >= 1.5 * args.bkd_size * args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)

        if len(remained_set) < len(data):
            for gid in cdd_ids:
                if bkd_num - len(picked_ids) <= 0:
                    break
                gsize = graph_sizes[gid]
                if gsize >= 1.0 * args.bkd_size * args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)

        picked_ids = list(set(picked_ids))
        remained_set = list(set(remained_set) - set(picked_ids))
        if len(remained_set) == 0 and bkd_num > len(picked_ids):
            print("no more graph to pick, return insufficient candidate graphs, try smaller bkd-pattern or graph size")

    return picked_ids
# return 1D list
def select_cdd_graphs(args, data: list, adj_list: list, subset: str,labels:list,sorted_ids=None):
    ''' clean_label_version
    Given a data (train/test), (randomly or determinately) 
    pick up some graph to put backdoor information, return ids.
    '''

    rs = np.random.RandomState(args.seed)
    graph_sizes = [np.array(adj).shape[0] for adj in adj_list]
    bkd_graph_ratio = args.bkd_gratio_train if subset == 'train' else args.bkd_gratio_test
    bkd_num = int(np.ceil(bkd_graph_ratio * len(data)))
    assert len(data) > bkd_num, "Graph Instances are not enough"


    if args.chose!='random':#如果不是随机选样本 而是根据置信度或者loss选
        if args.cleanlabel == 0:#是否区分clean_label 不区分就随便选
            # 修复：跳过节点数小于触发器大小的图，避免 candidate graph too small
            picked = []
            for gid in sorted_ids:
                if len(picked) >= bkd_num:
                    break
                if graph_sizes[gid] >= args.bkd_size * args.bkd_num_pergraph:
                    picked.append(gid)
            return picked
        else:
            picked_ids = []
            #print('according to con', sorted_ids)

            targetlabel_ids=[i for i in sorted_ids if labels[i] == args.target_class]
            print('数据集比例',len(data),len(targetlabel_ids),bkd_num)
            for gid in targetlabel_ids:
                if bkd_num - len(picked_ids) <= 0:
                    break
                if graph_sizes[gid] >= 1 * args.bkd_size * args.bkd_num_pergraph and  gid not in picked_ids:
                    picked_ids.append(gid)
            for gid in targetlabel_ids:
                if bkd_num - len(picked_ids) <= 0:
                    break
                if gid not in picked_ids:
                    picked_ids.append(gid)

            assert len(picked_ids)>=bkd_num,"目标类训练集不够"
            print('picked_ids',len(picked_ids))

            return picked_ids
    #随机选 看是否区分clean_label
    if args.cleanlabel==0 :#不区分就随便选
        return select_cdd_graphs_random(args, data, adj_list, subset)
    else:
       # picked_ids = []
    # print('according to con', sorted_ids)

        targetlabel_ids = [i for i in sorted_ids if labels[i] == args.target_class]
        print('数据集比例', len(data), len(targetlabel_ids), bkd_num)
        picked_ids=rs.choice(targetlabel_ids, bkd_num, replace=False)

        return picked_ids
    # picked_ids=[]
    # # Randomly pick up graphs as backdoor candidates from data
    # remained_set = copy.deepcopy(data)
    # loopcount = 0
    # while bkd_num-len(picked_ids) >0 and len(remained_set)>0 and loopcount<=50:
    #     loopcount += 1
    #
    #     cdd_ids = rs.choice(remained_set, bkd_num-len(picked_ids), replace=False)
    #     #print('挑选目标类')
    #     for gid in cdd_ids:
    #         if bkd_num-len(picked_ids) <=0:
    #             break
    #         gsize = graph_sizes[gid]
    #         #print('选择标签',gsize,labels[gid],args.target_class)
    #         if gsize >= 3*args.bkd_size*args.bkd_num_pergraph:#选择本不是/就是目标类的数据来中毒
    #             if args.cleanlabel==1 and labels[gid] == args.target_class:
    #                 picked_ids.append(gid)
    #             elif args.cleanlabel==-1 and labels[gid] != args.target_class:
    #                     picked_ids.append(gid)
    #
    #
    #     if len(remained_set)<len(data):
    #         for gid in cdd_ids:
    #             if bkd_num-len(picked_ids) <=0:
    #                 break
    #             gsize = graph_sizes[gid]
    #             if gsize >= 1.5*args.bkd_size*args.bkd_num_pergraph and gid not in picked_ids:
    #                 if args.cleanlabel == 1 and labels[gid] == args.target_class:
    #                     picked_ids.append(gid)
    #                 elif args.cleanlabel == -1 and labels[gid] != args.target_class:
    #                     picked_ids.append(gid)
    #
    #     if len(remained_set)<len(data):
    #         for gid in cdd_ids:
    #             if bkd_num-len(picked_ids) <=0:
    #                 break
    #             gsize = graph_sizes[gid]
    #             if gsize >= 1.0*args.bkd_size*args.bkd_num_pergraph and gid not in picked_ids :
    #                 if args.cleanlabel == 1 and labels[gid] == args.target_class:
    #                     picked_ids.append(gid)
    #                 elif args.cleanlabel == -1 and labels[gid] != args.target_class:
    #                     picked_ids.append(gid)

        #print(picked_ids)
        #for gid in picked_ids:
           #print(labels[gid],graph_sizes[gid])
        picked_ids = list(set(picked_ids))
        remained_set = list(set(remained_set) - set(picked_ids))
        if len(remained_set)==0 and bkd_num>len(picked_ids):
            print("no more graph to pick, return insufficient candidate graphs, try smaller bkd-pattern or graph size")

    return picked_ids
             

def select_cdd_nodes(args, graph_cdd_ids, adj_list,features=None,model=None):
    '''
    Given a graph instance, based on pre-determined standard,
    find nodes who should be put backdoor information, return
    their ids.

    return: same sequece with bkd-gids
            (1) a 2D list - bkd nodes under each graph
            (2) and a 3D list - bkd node groups under each graph
                (in case of each graph has multiple triggers)
    '''


    rs = np.random.RandomState(args.seed)
    
    # step1: find backdoor nodes
    picked_nodes = []  # 2D, save all cdd graphs

    for gid in graph_cdd_ids:
        node_ids = [i for i in range(len(adj_list[gid]))]
        assert len(node_ids)==len(adj_list[gid]), 'node number in graph {} mismatch'.format(gid)

        bkd_node_num =  int(args.bkd_num_pergraph*args.bkd_size)
        # print(len(adj_list[gid]))
        assert bkd_node_num <= len(adj_list[gid]), "error in SelectCddGraphs, candidate graph too small"

        #随机版本
        if args.pos=='random':
            cur_picked_nodes = rs.choice(node_ids, bkd_node_num, replace=False)
        else: #选择重要节点版本
            cur_picked_nodes=[]
            if args.pos == 'import':
                adj_tensor = torch.tensor(adj_list[gid])
                degrees = adj_tensor.sum(dim=1) # 对每一行求和，得到每个节点的度
                # 以度中心性先筛选一批节点 然后drop之后得到分数 选择最重要的个节点
                sorted_nodes = torch.argsort(degrees, descending=True)
                can_trigger_nodes = sorted_nodes[:bkd_node_num*2]
            #去掉度中心性第一步筛选
            else:#选择不重要的时候没有拿中心性去做处理优化
                can_trigger_nodes=[torch.tensor(i) for i in node_ids]


            model.eval()
            model.to(torch.device('cpu'))
            node_score={}
            for i in range(len(can_trigger_nodes)):
                node_index = can_trigger_nodes[i]
                modified_adj = np.copy(adj_list[gid])  
                modified_adj[node_index, :] = 0  
                modified_adj[:, node_index] = 0
                fetures = np.copy(features[gid])
                fetures[node_index] = np.zeros_like(fetures[node_index])
                mask_matrix = np.ones((1, len(modified_adj)))
                mask_matrix = np.expand_dims(mask_matrix, axis=-1)

                fetures = torch.tensor(np.expand_dims(fetures, axis=0), dtype=torch.float32)
                modified_adj = torch.tensor(np.expand_dims(modified_adj, axis=0), dtype=torch.float32)
                re_output = model(
                [fetures, modified_adj, torch.tensor(mask_matrix, dtype=torch.float32), args.target_class, gid])
                loss_fn = F.cross_entropy
                loss = loss_fn(re_output, torch.tensor(args.target_class), reduction='sum').item()

                node_score[node_index] = loss

            if args.pos == 'import':
                imp_score_idx = sorted(node_score.items(), key=lambda x: x[1], reverse=True)#降序

            elif args.pos == 'least':
                imp_score_idx = sorted(node_score.items(), key=lambda x: x[1])  # 升序

            for u, v in imp_score_idx:
                cur_picked_nodes.append(u.item())
                if len(cur_picked_nodes) == bkd_node_num or len(cur_picked_nodes)==len(imp_score_idx):
                #print('cur_picked_nodes',cur_picked_nodes)
                    break

        picked_nodes.append(cur_picked_nodes)




        
    # step2: match nodes
    assert len(picked_nodes)==len(graph_cdd_ids), "backdoor graphs & node groups mismatch, check SelectCddGraphs/SelectCddNodes"

    node_groups = [] # 3D, grouped trigger nodes
    for i in range(len(graph_cdd_ids)):    # for each graph, devide candidate nodes into groups
        gid = graph_cdd_ids[i]
        nids = picked_nodes[i]

        assert len(nids)%args.bkd_size==0.0, "Backdoor nodes cannot equally be divided, check SelectCddNodes-STEP1"

        # groups within each graph
        groups = np.array_split(nids, len(nids)//args.bkd_size)
        # np.array_split return list[array([..]), array([...]), ]
        # thus transfer internal np.array into list
        # store groups as a 2D list.
        groups = np.array(groups).tolist()
        node_groups.append(groups)

    assert len(picked_nodes)==len(node_groups), "groups of bkd-nodes mismatch, check SelectCddNodes-STEP2"

    return picked_nodes, node_groups
                           
    