import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np

class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self, negM, T, batchSize):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize*2).cuda()
        
    def forward(self, x, targets):
        batchSize = x.size(0)
        
        eps = 1e-6
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        #reordered_x = reordered_x.data
        #print('x:',x)
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_() +1e-6
        #pos= pos/max(pos)
        #print(torch.any(torch.isinf(pos)), pos)
        #print(torch.any(torch.isnan(pos)))
        #get all innerproduct, remove diag
        #print(x.shape)
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()+1e-6
        #all_prob = all_prob/max(all_prob)
        #print(all_prob.shape)
        all_prob = all_prob*(self.diag_mat+1e-6)
        #print(all_prob, self.diag_mat, self.negM)
        #all_prob = all_prob / max(all_prob)
        #print(all_prob)
        #print(torch.any(torch.isinf(all_prob)))
        #print(torch.any(torch.isnan(all_prob)))
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos
        
        #print('all.div:',torch.any(torch.isinf(all_div)))
        #print('all.div:',torch.any(torch.isnan(all_div)))
        #==================loss 1======================
        lnPmt = torch.div(pos, all_div+eps)  #选择pow来减弱
        #print(pos.shape, pos)
        #print('lnPmt:',torch.any(torch.isnan(lnPmt)))
        #print(reordered_x.shape, x.shape, pos.shape, all_prob.shape, lnPmt.shape)
        #torch.Size([128, 64]) torch.Size([128, 64]) torch.Size([128]) torch.Size([128, 128]) torch.Size([128])
        # negative probability
        #print(all_div)
        Pon_div = all_div.repeat(batchSize,1)
        lnPon = torch.div(all_prob, Pon_div.t()+eps)
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        #print(lnPon.shape)
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
    
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize
        
        #print('loss:', torch.any(torch.isnan(loss)))
        #==================loss 2================
        #lnPmt2 = torch.div(pos, all_div).pow(1)  #选择pow来减弱

        ## negative probability
        #Pon_div2 = all_div.repeat(batchSize,1)
        #lnPon2 = torch.div(all_prob, Pon_div2.t())
        #lnPon2 = (-lnPon2.add(-1)).pow(0)
        #
        ## equation 7 in ref. A (NCE paper)
        #lnPon2.log_()
        ## also remove the pos term
        #lnPon2 = lnPon2.sum(1) - (-lnPmt2.add(-1)).log_()
        ##print(lnPon.shape)
        #lnPmt2.log_()

        #lnPmtsum2 = lnPmt2.sum(0)
        #lnPonsum2 = lnPon2.sum(0)

        ## negative multiply m
    
        #lnPonsum2 = lnPonsum2 * self.negM
        #loss2 = - (lnPmtsum2 + lnPonsum2)/batchSize
        return loss
