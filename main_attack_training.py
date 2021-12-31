import argparse
import numpy as np
import torch
import random
import utils
from utils_polblogs import load_polblogs_data
from gcnModel import GCN
from FDGUAinterface import FDGUAinterface

parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--hidden', type=int, default=16, help='hidden ')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Rate')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight Decay')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--anchorsNum', type=int, default=6, help='anchorsNum')
parser.add_argument('--ATiter', type=int, default=10, help='ATiter')
parser.add_argument('--anchorPool', type=int, default=90, help='anchorPool')
parser.add_argument('--embedSize', type=int, default=20, help='embedSize')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

'''load data'''
if args.dataset!='polblogs':
    adj,features,labels,idx_train,idx_val,idx_test=utils.load_data(args.dataset)
else:
    adj,features,labels,idx_train,idx_val,idx_test=load_polblogs_data()
'''build model'''
numOfFeatures=features.shape[1]
numOfClasses=int(max(labels)+1)
surModel=GCN(numOfFeatures,args.hidden,numOfClasses,args.dropout,args.lr,args.weight_decay)
'''pretrain'''
surModel.fit(features,adj,labels,idx_train,idx_val,idx_test,reTrain=False,isSave=False)
features_norm=utils.preprocess_features(features)
for epoch in range(10):
    attackModel= FDGUAinterface(surModel, adj, features, labels, idx_train, idx_val, idx_test,dataset=args.dataset,anchorsNum=args.anchorsNum)
    sortIdx,perLabel=attackModel.attack(args.embedSize,args.anchorPool,args.ATiter)

    perAll = []
    for i in list(idx_train):
        if labels[i] != perLabel:
            perAll.append(i)
    surModel = surModel.to('cpu')
    surModel.eval()
    adj_norm = utils.normalize_adj_tensor(adj)
    sortIdx = [int(i) for i in sortIdx]
    for _ in range(10):
        surModel.eval()
        adjTemp = adj.clone().detach()
        flipR = torch.zeros_like(adj)
        batch=random.sample(idx_train, args.embedSize)
        for b in (batch):
            flipIdx = random.sample(sortIdx[:args.anchorPool], args.anchorsNum)
            flip = torch.zeros(adj.shape[0])
            flip[flipIdx] = 1
            flipR[b] = flip
            flipR[:,b]=flip.t()
        adjTemp = adjTemp + torch.mul(flipR, (
                torch.ones_like(adjTemp) - torch.eye(adjTemp.shape[0]) - 2 * adjTemp))
        adjTemp_norm = utils.normalize_adj_tensor(adjTemp)
        surModel.eval()
        surModel.fit(features, adjTemp.cpu(), labels, idx_train, idx_val, idx_test, reTrain=True,train_iters=args.ATiter)
    # set isSave=True to save robust model
    surModel.fit(features, adj, labels, idx_train, idx_val, idx_test, reTrain=True,train_iters=0,isSave=False,modelNum=epoch)
    acc=surModel.test_one(list(idx_train),features,adj)
    print ("acc_train:{:.2f}".format(float(acc)))

