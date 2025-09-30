import sys, os
from torch_geometric.loader import DataLoader
sys.path.append(os.path.abspath('../..'))
from utils.datareader import DataReader,GraphData2,GraphData3
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#修改特征判别器版本
from utils.mask import recover_mask
from trojan.prop import forwarding

from model.detector import Detector as GCN
from model.detector import MLPDetector
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


    
class GraphTrojanNet(nn.Module):
    def __init__(self, sq_dim, layernum=1, dropout=0.05):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(sq_dim, sq_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sq_dim, sq_dim))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, input, mask, thrd, 
                device=torch.device('cpu'),
                activation='relu', 
                for_whom='topo',
                binaryfeat=False):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        GW = GradWhere.apply

        bkdmat = self.layers(input)
        if activation=='relu': #feat应该也是sigmoid吧
            bkdmat = F.relu(bkdmat)
        elif activation=='sigmoid':
            bkdmat = torch.sigmoid(bkdmat)    # nn.Functional.sigmoid is deprecated
        elif activation=='tanh':
            bkdmat = torch.tanh(bkdmat)

        if for_whom == 'topo':  # not consider direct yet
            bkdmat = torch.div(torch.add(bkdmat, bkdmat.transpose(0, 1)), 2.0)#转对称矩阵
        if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
            bkdmat = GW(bkdmat, thrd, device)
        bkdmat = torch.mul(bkdmat, mask)#把mask中为0的元素都置为0


        return bkdmat
    

def train_gtn(args, model, toponet: GraphTrojanNet, featnet: GraphTrojanNet,
               pset, nset, topomasks, featmasks, 
               init_dr: DataReader, bkd_dr: DataReader, Ainputs, Xinputs,train_graphs,whichone):
    """
    All matrix/array like inputs should already in torch.tensor format.
    All tensor parameters or models should initially stay in CPU when
    feeding into this function.
    
    About inputs of this function:
    - pset/nset: gids in trainset
    - init_dr: init datareader, keep unmodified inside of each resampling
    - bkd_dr: store temp adaptive adj/features, get by  init_dr + GTN(inputs)
    """
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
        cpu = torch.device('cpu')
    
    init_As = init_dr.data['adj_list']
    init_Xs = init_dr.data['features']
    bkd_As = bkd_dr.data['adj_list']
    bkd_Xs = bkd_dr.data['features']
    
    nodenums = [len(adj) for adj in init_As]
    glabels = torch.LongTensor(init_dr.data['labels']).to(cuda)
    glabels[pset] = args.target_class#先改标签  触发器初始化initial的时候好像已经改过了
    allset = np.concatenate((pset, nset))
    
    optimizer_topo = optim.Adam(toponet.parameters(),
                       lr=args.gtn_lr,
                       weight_decay=5e-4)
    optimizer_feat = optim.Adam(featnet.parameters(),
                       lr=args.feat_lr,
                       weight_decay=5e-4)

    gtn_epochs=args.gtn_epochs
    feat_epochs=args.feat_epochs
    if whichone=='topo':
        feat_epochs=0
    else:
        gtn_epochs=0

    
    
    #----------- training topo generator -----------#
    toponet.to(cuda)
    model.to(cuda)
    topo_thrd = torch.tensor(args.topo_thrd).to(cuda)
    criterion = nn.CrossEntropyLoss()

    toponet.train()
    in_dim = len(bkd_Xs[0][0])
    out_dim = 2
    discriminator = GCN( in_dim,  64, out_dim)
    optimizer_de = optim.Adam(discriminator.parameters(),
                                lr=0.001,
                                weight_decay=5e-4)
    discriminator.to(cuda)
    model.train()
    for _ in tqdm(range(gtn_epochs), desc="training topology generator"):
        optimizer_topo.zero_grad()
        # generate new adj_list by dr.data['adj_list']
        for gid in pset:
            SendtoCUDA(gid, [init_As, Ainputs, topomasks])    # only send the used graph items to cuda
            rst_bkdA = toponet(
                Ainputs[gid], topomasks[gid], topo_thrd, cuda, args.topo_activation, 'topo')
            bkd_dr.data['adj_list'][gid] = torch.add(rst_bkdA[:nodenums[gid], :nodenums[gid]], init_As[gid])   # only current position in cuda
            SendtoCPU(gid, [init_As, Ainputs, topomasks])
        #添加异常检测损失

        gdata = GraphData3(bkd_dr, train_graphs,pset)
        testloader = DataLoader(gdata, batch_size=args.SIGNETbatch_size, shuffle=True)
        train_loss, n_samples = 0, 0
        loss_samples = []
        #训练判别器
        #loss_fn = F.nll_loss()
        train_loss= 0
        n_samples= 0
        for  data in testloader:
            data = data.to(cuda)  # data格式为[X, A, graph_support, nodenums, labels]
            optimizer_de.zero_grad()
            output = discriminator(data)
            output = output.squeeze() if output.dim() == 2 and output.size(1) == 1 else output
            loss = F.nll_loss(output, data.d_y)

            loss.backward()
            optimizer_de.step()

            train_loss += loss.item() * len(output)
            n_samples += len(output)
        #print('train_loss',train_loss/n_samples)
        #训练生成器
        loss = forwarding(args, bkd_dr, model, allset, criterion)
        total_anomaly_loss = 0
        for data in testloader:
            data = data.to(cuda)
            shadow_model_output = discriminator(data)

            batch_size = shadow_model_output.size(0)
            anomaly_loss = F.nll_loss(shadow_model_output,torch.zeros(batch_size, dtype=torch.long).to(cuda))

            total_anomaly_loss += anomaly_loss.item() * len(shadow_model_output)
        anomaly_loss=total_anomaly_loss / n_samples

        finalloss = args.alpha*loss + args.beta*anomaly_loss
        print('attack', loss.item() ,'anomaly',anomaly_loss, finalloss)#
        finalloss.backward()
        optimizer_topo.step()
        torch.cuda.empty_cache()
    discriminator.to(cpu)
    toponet.eval()
    toponet.to(cpu)
    model.to(cpu)

    for gid in pset:
        if args.gtn_epochs != 0:
            SendtoCPU(gid, [bkd_dr.data['adj_list']])
            bkd_dr.data['adj_list'][gid] = bkd_dr.data['adj_list'][gid].detach()

        else:
            bkd_dr.data['adj_list'][gid]=torch.tensor(bkd_dr.data['adj_list'][gid])
    del topo_thrd
    torch.cuda.empty_cache()



    #----------- training feat generator -----------#

    featnet.to(cuda)
    model.to(cuda)
    feat_thrd = torch.tensor(args.feat_thrd).to(cuda)
    criterion = nn.CrossEntropyLoss()
    # in_dim = len(bkd_Xs[0][0])
    # out_dim = 2
    discriminator2 = MLPDetector(in_dim, 64, out_dim)
    optimizer_de = optim.Adam(discriminator2.parameters(),
                              lr=0.01,
                              weight_decay=5e-4)
    discriminator2.to(cuda)
    featnet.train()
    model.train()
    for epoch in tqdm(range(feat_epochs), desc="training feature generator"):
        optimizer_feat.zero_grad()
        # # generate new features by dr.data['features']
        for gid in pset:
            SendtoCUDA(gid, [init_Xs, Xinputs, featmasks])  # only send the used graph items to cuda
            rst_bkdX = featnet(
                Xinputs[gid], featmasks[gid], feat_thrd, cuda, args.feat_activation, 'feat')
            bkd_dr.data['features'][gid] = torch.add(rst_bkdX[:nodenums[gid]], init_Xs[gid])   # only current position in cuda
            bkd_dr.data['features'][gid]=bkd_dr.data['features'][gid].to(cpu)

            SendtoCPU(gid, [init_Xs, Xinputs, featmasks])

        gdata = GraphData3(bkd_dr, train_graphs, pset)

        testloader = DataLoader(gdata, batch_size=args.SIGNETbatch_size, shuffle=True)
        train_loss, n_samples = 0, 0
        loss_samples = []
        # 训练判别器
        train_loss = 0
        n_samples = 0
        for data in testloader:
            data = data.to(cuda)  # data格式为[X, A, graph_support, nodenums, labels]
            optimizer_de.zero_grad()
            data=data.detach()
            output = discriminator2(data)
            output = output.squeeze() if output.dim() == 2 and output.size(1) == 1 else output
            loss = F.nll_loss(output, data.d_y)
            loss.backward()
            optimizer_de.step()
            train_loss += loss.item() * len(output)
            n_samples += len(output)
        #('train_loss',train_loss/n_samples)
        total_anomaly_loss = 0
        for data in testloader:
            data = data.to(cuda)
            shadow_model_output = discriminator2(data)
            batch_size = shadow_model_output.size(0)
            anomaly_loss = F.nll_loss(shadow_model_output, torch.zeros(batch_size, dtype=torch.long).to(cuda))
            total_anomaly_loss += anomaly_loss.item() * len(shadow_model_output)
        anomaly_loss = total_anomaly_loss / n_samples

        loss = forwarding(
            args, bkd_dr, model, allset,  criterion)
        finalloss = args.alpha2*loss   +  args.beta2* anomaly_loss
        print('attack', loss.item(), 'anomaly', anomaly_loss, finalloss)
        finalloss.backward()

        optimizer_feat.step()
        torch.cuda.empty_cache()
        
    featnet.eval()
    featnet.to(cpu)
    model.to(cpu)

    if feat_epochs != 0:
        for gid in pset:
            SendtoCPU(gid, [bkd_dr.data['features']])


    del feat_thrd
    torch.cuda.empty_cache()
    
    return toponet, featnet

#----------------------------------------------------------------
def SendtoCUDA(gid, items):
    """
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    cuda = torch.device('cuda')
    for item in items:
        item[gid] = torch.as_tensor(item[gid], dtype=torch.float32).to(cuda)
        
        
def SendtoCPU(gid, items):
    """
    Used after SendtoCUDA, target object must be torch.tensor and already in cuda.
    
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    
    cpu = torch.device('cpu')
    for item in items:
        item[gid] = item[gid].to(cpu)