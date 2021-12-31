import torch
import torch.nn.functional as F
import utils
import numpy as np
import random
from tqdm import tqdm
class FDGUAinterface():

    def __init__(self, model, adj, features,labels,idx_train,idx_val,idx_test,dataset,anchorsNum, device='cuda'):
        self.device = device
        self.surrogate=model
        self.oriAdj=adj.to(self.device)
        self.oriFeatures=features
        features = utils.preprocess_features(features.clone().detach())
        self.features=features.to(self.device)
        self.labels=labels.to(self.device)
        self.idx_train=idx_train
        self.idx_val=idx_val
        self.idx_test = idx_test
        self.dataset=dataset
        self.anchorsNum=anchorsNum

    def attack(self,embedSize,anchorPool,retrainIter,max_epoch=5):
        oriAdj=self.oriAdj
        features=self.features
        labels=self.labels
        if type(oriAdj) is not torch.Tensor:
            features, oriAdj,labels = utils.to_tensor(features, oriAdj,labels, device=self.device)
        else:
            features = features.to(self.device)
            oriAdj = oriAdj.to(self.device)
            labels = labels.to(self.device)

        self.surrogate=self.surrogate.to(self.device)
        self.surrogate.eval()

        adj_norm = utils.normalize_adj_tensor(oriAdj)
        predOutput = self.surrogate(features, adj_norm)
        predLabels = predOutput.max(1)[1].type_as(labels)
        predLabels[self.idx_train] = labels[self.idx_train]
        self.predLabels = predLabels
        file = open('results/AT_cora/' + str(embedSize) + '_' + str(anchorPool) +'_'+str(retrainIter)+ '.txt', 'a')
        train_list = random.sample(list(self.idx_train), len(self.idx_train))
        labelList = [[] for _ in range(int(self.labels.max()) + 1)]
        for i in train_list:
            labelList[int(self.labels[i])].append(i)
        target = 6
        while target < int(self.labels.max()) + 1:
            epoch = 0
            best_epoch = 0
            stop_ = 0
            flag = 0
            max_val_rate = 0.0
            best_sort_idx=None
            best_mix_matrix=None

            print("target:{}".format(target))
            file.write("target:{}\n".format(target))
            while epoch < max_epoch:
                initLabel = random.randint(0, int(self.labels.max()))
                while initLabel == target:
                    initLabel = random.randint(0, int(self.labels.max()))
                curNode = labelList[initLabel].pop()
                flipV = torch.zeros(oriAdj.shape[0]).to(self.device)
                check = self.testSingleNode(flipV, curNode)
                if check == True:
                    flipV = self.simple_dp(flipV, curNode, target)
                    flipV = torch.clip(flipV, 0, 1)
                    if self.predict(flipV,curNode)!=target:
                        file.write("simple dp fail ! try again !\n")
                        continue

                '''test continuousFlip in train set'''
                attackCountTrain = 0
                for i in (list(self.idx_train)):
                    check = self.testSingleNode(flipV, i)
                    if check == False:
                        attackCountTrain += 1
                print("epoch={},continuousFlip={:.4f},train fooling rate={:.4f}".format(epoch, flipV.sum(),attackCountTrain / len(list(self.idx_train))))
                file.write("epoch={},continuousFlip={:.4f},train fooling rate={:.4f}\n".format(epoch, flipV.sum(),attackCountTrain / len(list(self.idx_train))))

                flipV1,sortIdx = self.vote(flipV, train_list,target, self.anchorsNum)

                '''test anchors in val set'''
                attackCountVal = 0
                for i in tqdm(list(self.idx_val)):
                    check = self.testSingleNode(flipV1, i)
                    if check == False:
                        attackCountVal += 1
                cur_val_rate = attackCountVal / len(list(self.idx_val))
                print("epoch={},anchorsNum={:.4f},val fooling rate={:.4f}".format(epoch, flipV1.sum(), cur_val_rate))
                file.write("epoch={},anchorsNum={:.4f},val fooling rate={:.4f}\n".format(epoch, flipV1.sum(), cur_val_rate))

                '''test anchors in train set'''
                labelsNum = int(self.labels.max()) + 1
                mixMatrix = np.zeros([labelsNum, labelsNum], dtype=np.int)
                attackCountTrainD = 0
                for i in (list(self.idx_train)):
                    check = self.testSingleNode(flipV1, i)
                    if check == False:
                        attackCountTrainD += 1
                    orilabel = int(self.labels[i])
                    attackedlabel = int(self.predict(flipV1, i))
                    mixMatrix[orilabel][attackedlabel] += 1
                print("epoch={},anchorsNum={},train fooling rate={:.4f}".format(epoch, flipV1.sum(),attackCountTrainD / len(list(self.idx_train))))
                file.write("epoch={},anchorsNum={},train fooling rate={:.4f}\n".format(epoch, flipV1.sum(),attackCountTrainD / len(list(self.idx_train))))
                print(mixMatrix)
                mixText=''.join(str(i)+'\n' for i in mixMatrix)
                file.write(mixText)
                file.write("\n")

                if cur_val_rate <= max_val_rate:
                    stop_ += 1
                    if stop_ > 3:
                        flag = 1
                else:
                    stop_ = 0
                    max_val_rate = cur_val_rate
                    best_epoch = epoch
                    best_sort_idx=sortIdx
                    best_mix_matrix=mixMatrix

                '''test anchors in test set'''
                attackCountTestD = 0
                mixMatrix = np.zeros([labelsNum, labelsNum], dtype=np.int)
                for i in tqdm(self.idx_test):
                    check = self.testSingleNode(flipV1, i)
                    if check == False:
                        attackCountTestD += 1
                    orilabel = int(self.predLabels[i])
                    attackedlabel = int(self.predict(flipV1, i))
                    mixMatrix[orilabel][attackedlabel] += 1
                print("epoch={},anchorsNum={},test fooling rate={:.4f}".format(epoch, flipV1.sum(),attackCountTestD / len(self.idx_test)))
                file.write("epoch={},anchorsNum={},test fooling rate={:.4f}\n".format(epoch, flipV1.sum(),attackCountTestD / len(self.idx_test)))
                print(mixMatrix)
                mixText=''.join(str(i)+'\n' for i in mixMatrix)
                file.write(mixText)
                file.write("\n")

                epoch += 1
                if flag == 1:
                    break
            print("target:{},best_epoch:{}".format(target, best_epoch))
            target += 1
            file.flush()
        file.close()
        return best_sort_idx,np.argmax(np.sum(best_mix_matrix,axis=0))

    def calGrad(self,flip,i):
        flipV = flip.clone().detach().requires_grad_()
        flipR=torch.zeros_like(self.oriAdj)
        flipR[i]=flipV
        flipR[:,i]=flipV

        perturedAdj = self.oriAdj + torch.mul(flipR, (
                torch.ones_like(self.oriAdj) - torch.eye(self.oriAdj.shape[0], device=self.device) - 2 * self.oriAdj))
        numOfClasses = max(self.labels) + 1

        grad = []
        self.surrogate.eval()
        perturedAdj_norm = utils.normalize_adj_tensor(perturedAdj)

        output = self.surrogate(self.features, perturedAdj_norm)
        grad_mask = torch.zeros(numOfClasses).to(self.device)
        retain=True
        for c in range(numOfClasses):
            if c==numOfClasses-1:
                retain=False
            cls = torch.LongTensor(np.array(c).reshape(1)).cuda()
            grad_mask[cls]=1
            gradCls=torch.autograd.grad(torch.sum(torch.mul(output[i:i + 1],grad_mask)),flipV,retain_graph=retain)[0]
            grad_mask[cls]=0
            grad.append(gradCls.cpu().numpy())
        return torch.from_numpy(np.array(grad)).to(self.device),torch.squeeze(output[i:i+1].detach()).to(self.device)
    def proj_cur(self,flip, i,target):

        grad,output=self.calGrad(flip,i)
        ori_label=self.predLabels[i]

        f_c = output[target] - output[ori_label]
        w_c = grad[target] - grad[ori_label]
        dis=torch.abs(f_c) / torch.norm(w_c)

        delta_s=dis*(grad[target]-grad[ori_label])

        return delta_s
    def testSingleNode(self,flip,i):
        flipV = torch.clone(flip).to(self.device)
        flipR=torch.zeros_like(self.oriAdj)
        flipR[i]=flipV
        flipR[:,i]=flipV
        perturedAdj = self.oriAdj + torch.mul(flipR, (
                torch.ones_like(self.oriAdj) - torch.eye(self.oriAdj.shape[0], device=self.device) - 2 * self.oriAdj))
        self.surrogate.eval()
        perturedAdj_norm = utils.normalize_adj_tensor(perturedAdj)
        output = self.surrogate(self.features, perturedAdj_norm)
        if torch.argmax(output[i])==self.predLabels[i]:
            return True
        return False
    def predict(self,flip,i):
        flipV = torch.clone(flip).to(self.device)
        flipR = torch.zeros_like(self.oriAdj)
        flipR[i] = flipV
        flipR[:, i] = flipV
        perturedAdj = self.oriAdj + torch.mul(flipR, (
                torch.ones_like(self.oriAdj) - torch.eye(self.oriAdj.shape[0], device=self.device) - 2 * self.oriAdj))
        self.surrogate.eval()
        perturedAdj_norm = utils.normalize_adj_tensor(perturedAdj)
        output = self.surrogate(self.features, perturedAdj_norm)
        return torch.argmax(output[i])
    def vote(self,flip,train_list,target,eps):
        nonzero_flip=torch.where(flip>0,torch.ones_like(flip),torch.zeros_like(flip))
        attackedList=[]
        for i in train_list:
            check=self.testSingleNode(flip,i)
            if check==False:
                attackedList.append(i)
        posiGrad=torch.zeros_like(flip)
        negaGrad = torch.zeros_like(flip)
        for i in attackedList:
            grad,_=self.calGrad(flip,i)
            grad_p=grad[target]
            grad_n=torch.sum(grad,dim=0)-grad_p
            posiGrad=posiGrad+torch.where(nonzero_flip>0,grad_p,torch.zeros_like(grad_p))
            negaGrad=negaGrad+torch.where(nonzero_flip>0,grad_n,torch.zeros_like(grad_n))
        scores = torch.where(nonzero_flip > 0, posiGrad-negaGrad, torch.zeros_like(flip))
        _,sortIdx=torch.sort(scores,descending=True)
        print (sortIdx[:eps])
        flipR=torch.zeros_like(flip)
        flipR[sortIdx[:eps]]=1
        return flipR,sortIdx
    def simple_dp(self,flip,i,target, iter_max=20,eta=1.001):
        iter = 0
        cur_flip= flip.clone().detach().to(self.device)
        while iter < iter_max:
            delta_s = self.proj_cur(cur_flip, i,target)
            delta_s[i]=0
            cur_flip=torch.clip(cur_flip+eta*delta_s,0,1)
            if self.predict(cur_flip,i)==target:
                print ("simple dp succeed!!--step:{}".format(iter+1))
                break
            iter+=1
        return cur_flip


