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
from model.gin2 import GIN
from model.sagpool import GraphSAGE
from config import parse_args

def run(args):
    assert torch.cuda.is_available(), 'no GPU available'
    cpu = torch.device('cpu')
    cuda = torch.device('cuda:0')

    # load data into DataReader object
    dr = DataReader(args)




    loaders = {}
    for split in ['train', 'test']:
        if split=='train':
            gids = dr.data['splits']['train']
        else:
            gids = dr.data['splits']['test']
        gdata = GraphData(dr, gids)
        loader = DataLoader(gdata,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_batch)
        # data in loaders['train/test'] is saved as returned format of collate_batch()
        loaders[split] = loader
    print('train %d, test %d' % (len(loaders['train'].dataset), len(loaders['test'].dataset)))

    # prepare model
    in_dim = loaders['train'].dataset.num_features
    out_dim = loaders['train'].dataset.num_classes
    if args.model == 'gcn':
        model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'gat':
        model = GAT(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, num_head=args.num_head)
    elif args.model=='sage':
        model = GraphSAGE(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'gin':
        model = GIN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise NotImplementedError(args.model)

    if args.save_clean_model==False:
        print('不训练读取')
        save_path = args.clean_model_save_path
        save_path = os.path.join(save_path, '%s-%s-%s.t5' % (args.model, args.dataset, str(args.train_ratio)))
        checkpoint = torch.load(save_path)
        # 提取保存的信息
        model_state_dict = checkpoint['model']
        model.load_state_dict(model_state_dict)
        sorted_ids=checkpoint['sorted_ids']
        return dr, model,sorted_ids

    # print('\nInitialize model')
    # print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

    # training
    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
    
    model.to(cuda)
    for epoch in range(args.train_epochs):
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        loss_samples=[]
        for batch_id, data in enumerate(loaders['train']):
            
            for i in range(len(data)):
                data[i] = data[i].to(cuda)

            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            optimizer.zero_grad()
            output = model(data)
            if len(output.shape)==1:
                output = output.unsqueeze(0)
            loss = loss_fn(output, data[4])

            loss.backward()
            optimizer.step()
            scheduler.step()

            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            #挑选置信度样本

            if args.chose!='random' and epoch==args.train_epochs-1:
                for i in range(len(output)):
                    loss_samples.append({
                        'loss': loss.item(),  # 记录损失值
                        'output': output[i],  # 记录模型输出
                        'target': data[4][i],  # 记录目标标签.真实标签
                        'confidence':torch.softmax(output[i], dim=0),
                        'id':data[5][i].item()
                })
        loss_samples_sorted = sorted(loss_samples, key=lambda x: x['loss'], reverse=True)
        confidence_samples_sorted = sorted(loss_samples,
                                                   key=lambda x: x['confidence'][x['target']].item())  # 预测为目标类的置信度
        if args.chose=='con':
            sorted_ids = [sample['id'] for sample in confidence_samples_sorted]
        elif args.chose=='loss':
            sorted_ids = [sample['id'] for sample in loss_samples_sorted]
        else:
            sorted_ids=[]
        if args.train_verbose and (epoch % args.log_every == 0 or epoch == args.train_epochs - 1):
            print('Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (
                epoch + 1, loss.item(), train_loss / n_samples, time_iter / (batch_id + 1)))

        if (epoch + 1) % args.eval_every == 0 or epoch == args.train_epochs-1:
            model.eval()
            start = time.time()
            test_loss, correct, n_samples = 0, 0, 0
            for batch_id, data in enumerate(loaders['test']):
                for i in range(len(data)):
                    data[i] = data[i].to(cuda)
                # if args.use_org_node_attr:
                #     data[0] = norm_features(data[0])
                output = model(data)
                if len(output.shape)==1:
                    output = output.unsqueeze(0)
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                pred = predict_fn(output)

                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

            eval_acc = 100. * correct / n_samples
            print('Test set (epoch %d): Average loss: %.4f, Accuracy: %d/%d (%.2f%s) \tsec/iter: %.2f' % (
                epoch + 1, test_loss / n_samples, correct, n_samples, 
                eval_acc, '%', (time.time() - start) / len(loaders['test'])))
    
    model.to(cpu)
    
    if args.save_clean_model:
        save_path = args.clean_model_save_path
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, '%s-%s-%s.t5' % (args.model, args.dataset, str(args.train_ratio)))
        
        torch.save({
                    'model': model.state_dict(),
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'eval_acc': eval_acc,
                    'sorted_ids':sorted_ids
                    }, save_path)
        print('Clean trained GNN saved at: ', os.path.abspath(save_path))

    return dr, model,sorted_ids


if __name__ == '__main__':
    args = parse_args()
    run(args)