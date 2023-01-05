from __future__ import print_function
import torch.nn.functional as F
import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
def plotMatrixPoint(Mat, Label):
    """
    :param Mat: 二维点坐标矩阵
    :param Label: 点的类别标签
    :return:
   """
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, 5))
    colors = cmap[Label]
    #fig=plt.figure()
    x = Mat[:, 0]
    y = Mat[:, 1]
    plt.scatter(np.array(x), np.array(y),40 , c=colors, marker='o')  # scatter函数只支持array类型数据
    #return fig
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader,opt,  use_logit=True, is_norm=True, classifier='LR_DC'):
#def meta_test(net, testloader, use_logit=true, is_norm=true, classifier='proto', opt=None):
    net.eval()
    acc = []
    #print(net)
    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            # print(idx)
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            #print(net)
            #print(support_xs.shape)

            batch_size, _, channel, height, width = support_xs.size()
            support_xs = support_xs.view(-1, channel, height, width)
            query_xs = query_xs.view(-1, channel, height, width)

            if use_logit:
                #print(support_xs.shape)
                support_xs=net(support_xs)
                #x= torch.tensor(x)
                support_xs = torch.tensor([item.cpu().detach().numpy() for item in support_xs]).cuda() 
                support_xs = support_xs.squeeze()
                #print(support_xs.shape)
                support_features = support_xs.view(support_xs.size(0), -1)
                
                query_xs = net(query_xs)
                query_xs = torch.tensor([item.cpu().detach().numpy() for item in query_xs]).cuda() 
                query_xs = query_xs.squeeze()
                query_features = query_xs.view(query_xs.size(0), -1)
                # print(query_xs.shape)
                #support_features = net(support_xs).view(support_xs.size(0), -1)
                  
                #query_features = net(query_xs).view(query_xs.size(0), -1)
            else: 
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)
            #print(support_features.shape, query_features.shape)
            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            #  clf = SVC(gamma='auto', C=0.1)
            if classifier == 'LR':
                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
                #print(query_ys_pred)
            elif classifier == 'LR_DC':
                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                #support_mean = np.mean(support_features, axis=0)
                #support_cov = np.cov(support_features.T) 
                
                # ===== tsen ===
                #data_support = np.array(support_features)     
                #label_support = np.array(support_ys)
                #tsne = TSNE(n_components=2, init='pca', random_state=0)
                #data_support = tsne.fit_transform(data_support) 
                #plotMatrixPoint(data_support, label_support)
                #plt.show()
                # ===== ten ===
                #beta = 0.5 
                #support_features = np.power(support_features[:, ] ,beta)
                #query_features = np.power(query_features[:, ] ,beta)

                #support_mean = np.mean(support_features, 1)
                #print(support_mean.shape, query_features.shape)
                #query_features = query_features + 0.005
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
                #print(query_ys_pred)
                #print(support_mean.shape, support_cov.shape)


            elif classifier == 'SVM':
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
                                                          C=1,
                                                          kernel='linear',
                                                          decision_function_shape='ovr'))
                beta = 0.5 
                support_features = np.power(support_features[:, ] ,beta)
                query_features = np.power(query_features[:, ] ,beta)

                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'NN':
                beta = 0.5 
                support_features = np.power(support_features[:, ] ,beta)
                query_features = np.power(query_features[:, ] ,beta)

                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'Cosine':
                beta = 0.5 
                support_features = np.power(support_features[:, ] ,beta)
                query_features = np.power(query_features[:, ] ,beta)

                query_ys_pred = Cosine(support_features, support_ys, query_features)
            elif classifier == 'Proto':
                beta = 0.5 
                support_features = np.power(support_features[:, ] ,beta)
                query_features = np.power(query_features[:, ] ,beta)

                query_ys_pred = Proto(support_features, support_ys, query_features, query_ys ,opt)
            elif classifier == 'Proto_cosine':
                beta = 0.5 
                support_features = np.power(support_features[:, ] ,beta)
                query_features = np.power(query_features[:, ] ,beta)

                query_ys_pred = Proto_cosine(support_features, support_ys, query_features, opt)
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
            #print(query_ys, query_ys_pred)
    return mean_confidence_interval(acc)


def Proto(support, support_ys, query, query_ys, opt):
    """Protonet classifier"""
    #print(support_ys.shape, support.shape)   
    #opt.n_ways = 5
    #opt.n_shots =1
    #print(query.shape, support.shape, support_ys.shape)
    #print(query.shape, query_ys.shape)   

    # ===== tsen ===
    # data_support = np.array(support)     
    # label_support = np.array(support_ys)
    # plotMatrixPoint(data_support, label_support)
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # plotMatrixPoint(data_support, label_support)
    # plt.show()
    # ===== tsen ===


    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    support = support.mean(axis=3)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    logits = - ((query - support)**2).sum(-1)
    pred = np.argmax(logits, axis=-1)
    pred = np.reshape(pred, (-1,))
    #print(query.shape, support.shape,logits.shape, pred.shape)
    return pred
  
def Proto_cosine(support, support_ys, query, opt):
    """Protonet classifier"""
    #opt.n_ways = 5
    #opt.n_shots =1
    #print(query.shape, support.shape, support_ys.shape)
    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    # print('11:',support.shape)
    support = support.mean(axis=3)
    #print('22:',support.shape)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    support = torch.tensor(support)
    query = torch.tensor(query)
    support = support.repeat(1, 75, 1, 1)
    query = query.repeat(1,1, 5,1)
    query = F.normalize(query, p=2, dim=3, eps=1e-12)
    support = F.normalize(support, p=2, dim=3, eps=1e-12)           
    logits = torch.sum(query * support, dim=3)
    pred = np.argmax(logits, axis=-1)
    pred = np.reshape(pred, (-1,)) 
    #print(pred)
    return pred


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
