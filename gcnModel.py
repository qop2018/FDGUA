import math
import torch
import utils
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    GCN layer (https://arxiv.org/abs/1609.02907)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, lr, weight_decay):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    def lastLayer(self,x,adj):
        x = F.relu(self.gc1(x, adj))
        return x
    def fit(self,features, adj, labels, idx_train=None,idx_val=None,idx_test=None, reTrain=True,train_iters=500,isSave=False,modelNum=0):
        features = utils.preprocess_features(features)
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels)

        adj_norm = utils.normalize_adj_tensor(adj)

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels
        self.isSave=isSave
        self.modelNum=modelNum

        if reTrain==True:
            self._train(idx_train,idx_val,train_iters)
            self.test(idx_test)
        else:
            self.load_model()
            self.test(idx_test)
    def _train(self,idx_train,idx_val,train_iters):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            if i % 10 == 0:
                acc_train = utils.accuracy(output[idx_train], self.labels[idx_train])
                acc_val = utils.accuracy(output[idx_val], self.labels[idx_val])
                loss_train = F.nll_loss(output[idx_train], self.labels[idx_train])
                loss_val = F.nll_loss(output[idx_val], self.labels[idx_val])
                print('Epoch {}, train_loss: {:.4f}, train_acc:{:.4f} val_loss: {:.4f}, val_acc:{:.4f}'.\
                      format(i,loss_train.item(),acc_train,loss_val.item(),acc_val))
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train],self.labels[idx_train])
            loss_train.backward()
            optimizer.step()

        if self.isSave==True:
            #save preTraining model
            torch.save(self.state_dict(), 'coraModel/{}.pkl'.format(i))
            #save rubust model
            # torch.save(self.state_dict(), 'coraModel/{}.pkl'.format(self.modelNum))

    def test(self,idx_test):
        self.eval()
        output = self.forward(self.features, self.adj_norm)
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print('Test---- test loss: {:.4f}, test_acc:{:.4f}'.format(loss_test.item(), acc_test))
    def load_model(self):
        #load preTraining model
        files = glob.glob('coraModel/*.pkl')
        for file in files:
            self.load_state_dict(torch.load(file))
    def test_one(self,i,features,adj):
        self.eval()
        features = utils.preprocess_features(features)
        if type(adj) is not torch.Tensor:
            features, adj = utils.to_tensor(features, adj)
        adj_norm = utils.normalize_adj_tensor(adj)
        output=self.forward(features, adj_norm)
        acc_test = utils.accuracy(output[i], self.labels[i])
        return acc_test
