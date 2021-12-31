import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.spatial.distance import cosine
from scipy.sparse.linalg.eigen.arpack import eigsh
from torch.utils.data import DataLoader
import random


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
'''normalize_adj || is_sparse_tensor || to_scipy || sparse_mx_to_torch_sparse_tensor --> sparse'''
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False
def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx
def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    edges = nx_graph.edges()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset_str == 'citeseer':
        for i in range(labels.shape[0]):
            if np.array_equal(labels[i], np.zeros(labels.shape[1])):
                labels[i][0] = 1

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    return adj,features,labels,idx_train,idx_val,idx_test
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features=features.numpy()
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features=torch.from_numpy(features)
    return features
def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)
def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def degree(index,N):
    out = torch.zeros((N, ), device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)
def edge_drop_weights(oriAdj):
    # spMatrix=sp.coo_matrix(oriAdj.cpu().numpy())
    # col_index=torch.tensor(spMatrix.col,dtype=torch.int64)
    # deg=degree(col_index,oriAdj.shape[0])
    # deg_col=deg[col_index].to(torch.float32)
    # s_col=torch.log(deg_col)
    # weights=(s_col.max()-s_col)/(s_col.max()-s_col.mean())
    # return weights

    oriAdj=torch.abs(oriAdj)
    oriAdjScore=torch.sum(oriAdj,dim=0)
    oriAdjScore=(oriAdjScore.max()-oriAdjScore)/(oriAdjScore.max()-oriAdjScore.mean())

    return oriAdjScore
def drop_edge_weighted(oriAdj,edge_weights,p=0.005,threshold=1.):
    spMatrix = sp.coo_matrix(oriAdj.cpu().numpy())
    edge_index = torch.stack((torch.tensor(spMatrix.row, dtype=torch.int64), torch.tensor(spMatrix.col, dtype=torch.int64)))

    edge_weights=edge_weights/edge_weights.mean()*p
    edge_weights=edge_weights.where(edge_weights<threshold,torch.ones_like(edge_weights)*threshold)
    sel_mask=torch.bernoulli(1.0-edge_weights).to(torch.bool)

    edge_index_mask=edge_index[:,sel_mask]
    data_mask=torch.tensor(spMatrix.data)[sel_mask]
    maskAdj=torch.sparse.FloatTensor(edge_index_mask,data_mask,
                                     torch.Size([oriAdj.shape[0],oriAdj.shape[1]])).to_dense()
    maskAdj=maskAdj.to(oriAdj.device)

    '''generate mask'''
    mask=oriAdj-maskAdj
    mask=torch.ones_like(oriAdj)-mask
    return mask
def feature_drop_weights(features):

    features=torch.abs(features)
    featureScore=torch.sum(features,dim=0)
    featureWeight=(featureScore.max()-featureScore)/(featureScore.max()-featureScore.mean())

    return featureWeight
def augment_features(w,feaVec,p=0.05,threshold=1.0):

    '''random'''
    # ind=torch.randint(0,feaVec.shape[0],[p])
    # ind=ind.to(feaVec.device)
    #
    # mask=torch.zeros(feaVec.shape[0])
    # mask=mask.to(feaVec.device)
    # mask[ind]=1
    #
    # augF=torch.clone(feaVec)
    # augF[ind]=-(augF[ind]-mask[ind])
    # return augF

    w = torch.exp(w / w.mean()) * p
    drop_prob = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = torch.bernoulli(drop_prob)

    augF = feaVec.clone().detach()
    augF=torch.where(drop_mask==1,-(augF-drop_mask),augF)

    return augF
def augment_edges(w,edgeVec,p=0.05,threshold=1.0):

    w = torch.exp(w / w.mean()) * p
    drop_prob = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = torch.bernoulli(drop_prob)

    augE = edgeVec.clone().detach()
    augE=torch.where(drop_mask==1,-(augE-drop_mask),augE)

    return augE
def delete_edges(adjVec,p):
    augVec=adjVec.clone().detach()

    oriLinkIdx=np.nonzero(((adjVec==1).int().cpu().numpy()))[0]
    delNum=((adjVec==1).sum()*p).floor()
    delIdx=random.sample(list(oriLinkIdx),int(delNum))
    augVec[delIdx]=0
    return augVec

def add_edges(adjVec,oriLabel,predLabels,p):
    augVec = adjVec.clone().detach()

    proLinkIdx=np.nonzero(((predLabels == oriLabel).int().cpu().numpy()))[0]
    addNum = ((adjVec == 1).sum() * p).floor()
    addIdx = random.sample(list(proLinkIdx), int(addNum))
    augVec[addIdx] = 1
    return augVec
def replace_edges(adjVec,oriLabel,predLabels,p):
    augVec = adjVec.clone().detach()

    oriLinkIdx = np.nonzero(((adjVec == 1).int().cpu().numpy()))[0]
    delNum = ((adjVec == 1).sum() * p).floor()
    delIdx = random.sample(list(oriLinkIdx), int(delNum))
    augVec[delIdx] = 0

    proLinkIdx = np.nonzero(((predLabels == oriLabel).int().cpu().numpy()))[0]
    addNum = ((adjVec == 1).sum() * p).floor()
    addIdx = random.sample(list(proLinkIdx), int(addNum))
    augVec[addIdx] = 1

    return augVec

def selectNode(features,adj,labels,idx_test,surrogate):
    surrogate.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = surrogate(preprocess_features(features), adj_norm)

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    '''modify'''
    return high + low + other
    # return other
def selectWorstNode(features,adj,labels,idx_test,surrogate):
    surrogate.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = surrogate(preprocess_features(features), adj_norm)
    highestMargin=0
    node=0
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin>highestMargin:
            highestMargin=margin
            node=idx
    return node

def classification_margin(output, true_label):

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def bisection(a, eps, xi, ub=1):
    pa = np.clip(a, 0, ub)
    if np.sum(pa) <= eps:
        # print('np.sum(pa) <= eps !!!!')
        upper_S_update = pa
    else:
        mu_l = np.min(a - 1)
        mu_u = np.max(a)
        # mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l) > xi:
            # print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l) / 2
            gu = np.sum(np.clip(a - mu_a, 0, ub)) - eps
            gu_l = np.sum(np.clip(a - mu_l, 0, ub)) - eps
            # print('gu:',gu)
            if gu == 0:
                print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a

        upper_S_update = np.clip(a - mu_a, 0, ub)

    return torch.from_numpy(upper_S_update)

