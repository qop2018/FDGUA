import argparse
import numpy as np
import torch
import random
import utils
from utils_polblogs import load_polblogs_data
from gcnModel import GCN
from FDGUA import FDGUA

parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--hidden', type=int, default=16, help='hidden ')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Rate')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight Decay')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--anchorsNum', type=int, default=6, help='anchorsNum')

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

'''attack'''
attackModel=FDGUA(surModel,adj,features,labels,idx_train,idx_val,idx_test,anchorsNum=args.anchorsNum,dataset=args.dataset)
attackModel.attack()
