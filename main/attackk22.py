import sys, os
import datetime
from pathlib import Path
# 加入 异常检测loss

s_path = str(Path(__file__).parent.parent)  # 从B文件夹向上回退3层到S
print(s_path)
sys.path.append(s_path)

import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from utils.datareader import DataReader, GraphData2, GraphData3
from utils.bkdcdd import select_cdd_graphs, select_cdd_nodes, select_cdd_graphs_random
from utils.mask import gen_mask, recover_mask
import main.benign as benign
import trojan.GTAA22 as gta
from trojan.input import gen_input, custom_collate_fn
from trojan.prop import train_model, evaluate
from config import parse_args

# from anomaly.detection import *


import pprint
from torch_geometric.loader import DataLoader


class GraphBackdoor:
    def __init__(self, args) -> None:
        self.args = args

        assert torch.cuda.is_available(), 'no GPU available'
        self.cpu = torch.device('cpu')
        self.cuda = torch.device('cuda')

    def run(self):
        # train a benign GNN
        self.benign_dr, self.benign_model, sorted_ids = benign.run(self.args)  # 数据集和干净模型

        # return

        model = copy.deepcopy(self.benign_model).to(self.cuda)
        # pick up initial candidates
        bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = self.bkd_cdd('test', model)
        nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
        nodemax = max(nodenums)
        featdim = np.array(self.benign_dr.data['features'][0]).shape[1]

        # init two generators for topo/feat 两个特征生成
        toponet = gta.GraphTrojanNet(nodemax, self.args.gtn_layernum)
        featnet = gta.GraphTrojanNet(featdim, self.args.gtn_layernum)

        # init test data
        # NOTE: for data that can only add perturbation on features, only init the topo value
        init_dr_test = self.init_trigger(
            self.args, copy.deepcopy(self.benign_dr), bkd_gids_test, bkd_nid_groups_test, 0.0, 0.0)
        bkd_dr_test = copy.deepcopy(init_dr_test)

        topomask_test, featmask_test = gen_mask(
            init_dr_test, bkd_gids_test, bkd_nid_groups_test)

        Ainput_test, Xinput_test = gen_input(self.args, init_dr_test, bkd_gids_test)

        # ----------- training anomaly detection -----------#这部分删掉 是之前的版本
        # 用目标类训练一个异常检测 然后去捕捉看起来不属于目标类的图（trigger图）
        train_graphs=self.benign_dr.data['splits']['train']
        target_graphs=[]
        #for data in self.benign_dr:
        for gid in train_graphs:
            if self.benign_dr.data['labels'][gid]==args.target_class:
                target_graphs.append(gid)
        # train_graphs2 = self.benign_dr.data['splits']['test']
        # target_graphs2 = []
        # # for data in self.benign_dr:
        # for gid in train_graphs2:
        #     if self.benign_dr.data['labels'][gid] == args.target_class:
        #         target_graphs2.append(gid)
        # #target_graphs=list(target_graphs)
        # #print(len(target_graphs))
        # gdata = GraphData2(self.benign_dr, target_graphs)
        #
        #
        # dataloader = DataLoader(gdata, batch_size=args.SIGNETbatch_size, shuffle=True)
        #
        # device=self.cuda
        # detectionmodel = SIGNET(featdim, 0, args, device).to(device)
        # optimizer = torch.optim.Adam(detectionmodel.parameters(), lr=args.SIGNET_lr)
        # save_path = args.clean_model_save_path
        # save_path = os.path.join(save_path,
        #                          '%s-%s-%s.detection' % (args.dataset, str(args.epochs), str(args.SIGNET_lr)))
        # if args.readdetection】 and os.path.exists(save_path):
        #     checkpoint = torch.load(save_path)
        #     model_state_dict = checkpoint['model']
        #     model.load_state_dict(model_state_dict)
        # else:
        #     for epoch in range(1, args.epochs + 1):
        #
        #         detectionmodel.train()
        #         loss_all = 0
        #         num_sample = 0
        #         for data in dataloader:
        #             optimizer.zero_grad()
        #             data = data.to(device)
        #             y, y_hyper, node_imp, edge_imp = detectionmodel(data)
        #             #print('y',y,y_hyper)
        #             loss = detectionmodel.loss_nce(y, y_hyper).mean()  # 这里的y是自己生成的不是外部标签
        #             #print('loss',loss)
        #             loss_all += loss.item() * data.num_graphs
        #             num_sample += data.num_graphs
        #             loss.backward()
        #             optimizer.step()
        #
        #
        #         info_train = 'Epoch {:3d}, Loss CL {:.4f}'.format(epoch, loss_all / num_sample)
        #         print('traindetection:',info_train)
        # #保存这个数据集的detectionmodel以免每次都重新训练
        #     torch.save({
        #         'model': model.state_dict()
        #
        #     }, save_path)
        for rs_step in range(self.args.resample_steps):  # for each step, choose different sample 默认是选3次

            # randomly select new graph backdoor samples随机选择后门样本  训练集每次选的都不一样
            bkd_gids_train, bkd_nids_train, bkd_nid_groups_train = self.bkd_cdd('train', model, sorted_ids)

            # positive/negtive sample set  正负样本并不更新
            pset = bkd_gids_train  # 正样本就是训练集中被选中插入trigger的图 也就是要被异常检测找出来的图
            nset = list(set(self.benign_dr.data['splits']['train']) - set(pset))  # 负样本就是剩下的干净的训练集

            if self.args.pn_rate != None:
                if len(pset) > len(nset):
                    repeat = int(np.ceil(len(pset) / (len(nset) * self.args.pn_rate)))
                    nset = list(nset) * repeat
                else:
                    repeat = int(np.ceil((len(nset) * self.args.pn_rate) / len(pset)))
                    pset = list(pset) * repeat  # 用于generator的训练 不影响训练集

            # init train data
            # NOTE: for data that can only add perturbation on features, only init the topo value
            init_dr_train = self.init_trigger(
                self.args, copy.deepcopy(self.benign_dr), bkd_gids_train, bkd_nid_groups_train, 0.0, 0.0)
            bkd_dr_train = copy.deepcopy(init_dr_train)

            topomask_train, featmask_train = gen_mask(
                init_dr_train, bkd_gids_train, bkd_nid_groups_train)
            Ainput_train, Xinput_train = gen_input(self.args, init_dr_train, bkd_gids_train)

            oldrate = 90
            for bi_step in range(self.args.bilevel_steps):
                print("Resampling step %d, bi-level optimization step %d" % (rs_step, bi_step))
                if args.readtrigger == True:
                    save_path = os.path.join(self.args.bkd_model_save_path, '%s-%s-%f-%.2f-%.2f-%.2f-%.2f.t5' % (
                        self.args.model, self.args.dataset, rs_step, self.args.train_ratio,
                        self.args.save_gratio_train, self.args.bkd_num_pergraph, self.args.bkd_size))

                    checkpoint = torch.load(save_path)
                    # 提取保存的信息

                    toponet_state_dict = checkpoint['toponet']
                    toponet.load_state_dict(toponet_state_dict)
                    featnet_state_dict = checkpoint['featnet']
                    featnet.load_state_dict(featnet_state_dict)

                else:
                    toponet, _ = gta.train_gtn(
                        self.args, model, toponet, featnet,
                        pset, nset, topomask_train, featmask_train,
                        init_dr_train, bkd_dr_train, Ainput_train, Xinput_train, train_graphs, 'topo')

                # get new backdoor datareader for training based on well-trained generators
                for gid in bkd_gids_train:  # 更新了训练集的topo
                    if args.notopo:
                        bkd_dr_train.data['adj_list'][gid] = torch.tensor(init_dr_train.data['adj_list'][gid])
                    else:
                        rst_bkdA = toponet(Ainput_train[gid], topomask_train[gid], self.args.topo_thrd, self.cpu,
                                           self.args.topo_activation, 'topo')
                        bkd_dr_train.data['adj_list'][gid] = torch.add(
                            rst_bkdA[:nodenums[gid], :nodenums[gid]].detach().cpu(),
                            torch.tensor(init_dr_train.data['adj_list'][gid]))

                train_model(self.args, bkd_dr_train, model, list(set(pset)), list(set(nset)))  # 学完topo train一下

                # 保存topo这一步训练集
                bkd_path = os.path.join(self.args.bkd_data_save_path,
                                        args.dataset + 'new22aftopotrain_' + args.model + str(
                                            args.cleanlabel) + '_' + str(
                                            args.bkd_size) + '_' + str(args.beta) + '_' + str(args.alpha) + '_' + str(
                                            args.alpha2) + args.chose + '_' + args.pos + '_' + str(
                                            args.bkd_gratio_train) + '_' + str(bi_step))

                # print('traintopodata is saved at', bkd_path)
                # torch.save({'bkd_dr_train': bkd_dr_train, 'bkd_gids_train': bkd_gids_train, 'benign_dr': self.benign_dr,
                #           'bkd_dr_test': bkd_dr_test, 'bkd_gids_test': bkd_gids_test, 'pset': pset}, bkd_path)

                # ----------------- Evaluation -----------------#
                for gid in bkd_gids_test:  # 更新测试集的topo
                    rst_bkdA = toponet(
                        Ainput_test[gid], topomask_test[gid], self.args.topo_thrd,
                        self.cpu, self.args.topo_activation, 'topo')
                    bkd_dr_test.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]],
                        torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
                bkd_path = os.path.join(self.args.bkd_data_save_path,
                                        args.dataset + 'new22aftopotest_' + args.model + str(
                                            args.cleanlabel) + '_' + str(
                                            args.bkd_size) + '_' + str(args.beta) + '_' + str(args.alpha) + '_' + str(
                                            args.alpha2) + args.chose + '_' + args.pos + '_' + str(
                                            args.bkd_gratio_train) + '_' + str(bi_step))
                # print('testtopodata is saved at', bkd_path)
                # torch.save({'bkd_dr_train': bkd_dr_train, 'bkd_gids_train': bkd_gids_train, 'benign_dr': self.benign_dr,
                #           'bkd_dr_test': bkd_dr_test, 'bkd_gids_test': bkd_gids_test, 'pset': pset}, bkd_path)

                if args.readtrigger != True:
                    _, featnet = gta.train_gtn(
                        self.args, model, toponet, featnet,
                        pset, nset, topomask_train, featmask_train,
                        init_dr_train, bkd_dr_train, Ainput_train, Xinput_train, train_graphs, 'feat')

                for gid in bkd_gids_train:  # 更新了训练集的feature
                    if args.nofeat:
                        bkd_dr_train.data['features'][gid] = torch.tensor(init_dr_train.data['features'][gid])
                    else:
                        rst_bkdX = featnet(
                            Xinput_train[gid], featmask_train[gid], self.args.feat_thrd,
                            self.cpu, self.args.feat_activation, 'feat')
                        bkd_dr_train.data['features'][gid] = torch.add(
                            rst_bkdX[:nodenums[gid]].detach().cpu(),
                            torch.tensor(init_dr_train.data['features'][gid]))  # 更新特征矩阵 特征相加

                train_model(self.args, bkd_dr_train, model, list(set(pset)), list(set(nset)))  # 学完feattrain一下

                # ----------------- Evaluation -----------------#

                for gid in bkd_gids_test:  # 更新测试集的feature
                    rst_bkdX = featnet(
                        Xinput_test[gid], featmask_test[gid], self.args.feat_thrd,
                        self.cpu, self.args.feat_activation, 'feat')
                    bkd_dr_test.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]], torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))

                # graph originally in target label
                yt_gids = [gid for gid in bkd_gids_test
                           if self.benign_dr.data['labels'][gid] == self.args.target_class]
                # graph originally notin target label
                yx_gids = list(set(bkd_gids_test) - set(yt_gids))
                clean_graphs_test = list(set(self.benign_dr.data['splits']['test']) - set(bkd_gids_test))

                # feed into GNN, test success rate
                bkd_acc = evaluate(self.args, bkd_dr_test, model, bkd_gids_test)  # 所有的中毒样本
                flip_rate = evaluate(self.args, bkd_dr_test, model, yx_gids)  # 要被攻击翻转的样本（也就是原来非target的中毒样本）
                clean_acc = evaluate(self.args, bkd_dr_test, model, clean_graphs_test)  # 测试集里的干净图（用来测CAD的）

                # save gnn rs_step == 0 and
                if True:  # (bi_step==self.args.bilevel_steps-1) or (abs(bkd_acc-100) <1e-4) :# or (flip_rate-oldrate<0.005 and flip_rate-oldrate>0):
                    if self.args.save_bkd_model:
                        print('save at', bi_step, 'step')

                        save_path = self.args.bkd_model_save_path
                        os.makedirs(save_path, exist_ok=True)
                        save_path = os.path.join(save_path, '%s-%s-%.2f-%.2f-%.2f-%.2f-%.2f.t5' % (
                            self.args.model, self.args.dataset, rs_step, self.args.train_ratio,
                            self.args.bkd_gratio_train, self.args.bkd_num_pergraph, self.args.bkd_size))
                        if args.readtrigger == False:
                            torch.save({'model': model.state_dict(),
                                        'toponet': toponet.state_dict(),
                                        'featnet': featnet.state_dict(),
                                        'asr': bkd_acc,
                                        'flip_rate': flip_rate,
                                        'clean_acc': clean_acc,
                                        }, save_path)
                        print('asr', bkd_acc, 'flip_rate', flip_rate, 'clean_acc', clean_acc)
                        print("Trojaning model and generator is saved at: ", save_path)
                        # 在这里保存最终的中毒训练集用作分析和异常检测
                        print('最终的后门训练集', bkd_gids_train)  # 后门数据集和注入的图列表
                        # bkd_gids_train, bkd_nids_train, bkd_nid_groups_train

                        if args.readtrigger:
                            bkd_path = os.path.join(self.args.bkd_data_save_path,
                                                    args.dataset + '_' + args.model + str(args.cleanlabel) + '_' + str(
                                                        args.bkd_size) + '_' + str(
                                                        args.beta) + '_' + args.chose + '_' + args.pos + '_retest' + str(
                                                        args.bkd_gratio_train))
                        else:
                            if args.nofeat:
                                bkd_path = os.path.join(self.args.bkd_data_save_path,
                                                        args.dataset + 'nofeat22_' + args.model + str(
                                                            args.cleanlabel) + '_' + str(args.bkd_size) + '_' + str(
                                                            args.beta) + '_' + str(
                                                            args.alpha) + '' + args.chose + '_' + args.pos + '_' + str(
                                                            args.bkd_gratio_train) + '_' + str(bi_step))
                            elif args.notopo:
                                bkd_path = os.path.join(self.args.bkd_data_save_path,
                                                        args.dataset + 'notopo_' + args.model + str(
                                                            args.cleanlabel) + '_' + str(args.bkd_size) + '_' + str(
                                                            args.beta2) + '_' + str(
                                                            args.alpha2) + '' + args.chose + '_' + args.pos + '_' + str(
                                                            args.bkd_gratio_train) + '_' + str(bi_step))
                            else:
                                bkd_path = os.path.join(self.args.bkd_data_save_path,
                                                        args.dataset + 'new22_' + args.model + str(
                                                            args.cleanlabel) + '_' + str(args.bkd_size) + '_beta' + str(
                                                            args.beta) + '_2beta' + str(args.beta2) + '_a' + str(
                                                            args.alpha) + '_2a' + str(
                                                            args.alpha2) + args.chose + '_' + args.pos + '_' + str(
                                                            args.bkd_gratio_train) + '_' + str(bi_step))
                        print('bkddata is saved at', bkd_path)
                        torch.save({'bkd_dr_train': bkd_dr_train, 'bkd_gids_train': bkd_gids_train,
                                    'benign_dr': self.benign_dr,
                                    'bkd_dr_test': bkd_dr_test, 'bkd_gids_test': bkd_gids_test, 'pset': pset}, bkd_path)
                        # torch.save({'bkd_dr_test': bkd_dr_test, 'bkd_gids_test': bkd_gids_test,
                        #             'bkd_nid_groups_test': bkd_nid_groups_test}, bkd_path)

                        # for gid in bkd_gids_train:

                if flip_rate > oldrate:
                    oldrate = flip_rate

                if abs(bkd_acc - 100) < 1e-4:  # 攻击成功了就结束
                    # bkd_dr_tosave = copy.deepcopy(bkd_dr_test)
                    # print('最终的后门训练集', bkd_dr_train)
                    print("Early Termination for 100% Attack Rate")
                    if args.nostop == False:
                        break
                if (flip_rate - oldrate < 0.005 and flip_rate - oldrate > 0):
                    print("Early Termination for no improve")
                    # break
        print('Done')

    def bkd_cdd(self, subset: str, model=None, sorted_ids=None):
        # - subset: 'train', 'test'
        # find graphs to add trigger (not modify now)
        if subset == 'test':
            bkd_gids = select_cdd_graphs_random(self.args, self.benign_dr.data['splits'][subset],
                                                self.benign_dr.data['adj_list'], subset)
        else:
            bkd_gids = select_cdd_graphs(
                self.args, self.benign_dr.data['splits'][subset], self.benign_dr.data['adj_list'], subset,
                self.benign_dr.data['labels'], sorted_ids)
        # find trigger nodes per graph
        # bkd_gids = select_cdd_graphs(
        #     self.args, self.benign_dr.data['splits'][subset], self.benign_dr.data['adj_list'], subset,self.benign_dr.data['labels'])

        # same sequence with selected backdoored graphs
        bkd_nids, bkd_nid_groups = select_cdd_nodes(
            self.args, bkd_gids, self.benign_dr.data['adj_list'], self.benign_dr.data['features'], model)

        assert len(bkd_gids) == len(bkd_nids) == len(bkd_nid_groups)

        return bkd_gids, bkd_nids, bkd_nid_groups  # 要攻击的图 候选节点 按triggersize分组的候选节点

    @staticmethod
    def init_trigger(args, dr: DataReader, bkd_gids: list, bkd_nid_groups: list, init_edge: float, init_feat: float):
        if init_feat == None:
            init_feat = - 1
            print('init feat == None, transferred into -1')

        # (in place) datareader trigger injection 连接全断开 邻接矩阵对应边置为0
        for i in tqdm(range(len(bkd_gids)), desc="initializing trigger..."):
            gid = bkd_gids[i]
            for group in bkd_nid_groups[i]:
                # change adj in-place
                src, dst = [], []
                for v1 in group:
                    for v2 in group:
                        if v1 != v2:
                            src.append(v1)
                            dst.append(v2)
                a = np.array(dr.data['adj_list'][gid])
                a[src, dst] = init_edge
                dr.data['adj_list'][gid] = a.tolist()

                # change features in-place 对应节点修改为全0
                featdim = len(dr.data['features'][0][0])
                a = np.array(dr.data['features'][gid])
                a[group] = np.ones((len(group), featdim)) * init_feat
                dr.data['features'][gid] = a.tolist()

            # change graph labels
            assert args.target_class is not None
            dr.data['labels'][gid] = args.target_class

        return dr


if __name__ == '__main__':
    args = parse_args()
    command_line = " ".join(sys.argv)
    print(command_line)
    start_time = datetime.datetime.now()
    attack = GraphBackdoor(args)
    attack.run()
    end_time = datetime.datetime.now()
    print("执行时间：{}".format(end_time - start_time))
