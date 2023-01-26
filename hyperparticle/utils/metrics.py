import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import networkx as nx
from random import randint
from typing import Generator, Tuple
from graphicle import Graphicle
from numpy import ndarray

from .visualisation import hard_descendants
from sklearn.metrics import precision_score, recall_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import ot

def sqdist(x, y):
    sq_norm_x = np.linalg.norm(x, axis=-1) ** 2.
    sq_norm_y = np.linalg.norm(y, axis=-1) ** 2.
    sq_norm_xy = np.linalg.norm(x - y, axis=-1) ** 2.

    cosh_angle = 1 + 2 * sq_norm_xy / ((1 - sq_norm_x) * (1 - sq_norm_y))
    cosh_angle = np.clip(a=cosh_angle, a_min=1. + 1e-8, a_max=None)
    dist = np.arccosh(cosh_angle)
    return dist #** 2.


def distance_matrix(nodes, off_diag = True):
    length = len(nodes)
    matrix = np.zeros((length, length))
    for n_idx in range(length):
        nd = nodes[n_idx][None, :]
        matrix[n_idx] = sqdist(nd, nodes) + 1e-8

    if off_diag == True:
        return matrix[np.triu_indices(n=length, k=1)]
    else:
        return matrix


def precision_and_recall(event):
    '''Get precision and recall for all the different graphs.
    '''
    keys = [k for k in event.keys()]
    precision = []
    recall = []

    k = 0
    graph, hyp = event[keys[k]], event[keys[k+1]]
    hard_mask = hard_descendants(graph, 25)

    
    y_true = hard_mask[graph.final.data].astype(int)
    
    hyp_final = hyp[graph.final.data]
    dist = distance_matrix(hyp_final, off_diag=False)
    y_pred = np.array([y_true[np.argsort(j)[1]] for j in dist])

    precision.append(precision_score(y_true, y_pred))
    recall.append(recall_score(y_true, y_pred))

    for k in [2, 4, 6]:
        graph, hyp = event[keys[k]], event[keys[k+1]]
        hyp_final = hyp[graph.final.data, :]
        dist = distance_matrix(hyp_final, off_diag=False)
        y_pred = np.array([y_true[np.argsort(j)[1]] for j in dist])

        precision.append(precision_score(y_true, y_pred))
        recall.append(recall_score(y_true, y_pred))

    return precision, recall


def mAP(graph, hyp ):
    '''Get the mean average precision for all the different graphs.
    '''
    G = nx.Graph()
    G.add_edges_from(graph.edges)
    distances = distance_matrix(hyp, off_diag=False)
    mAP = 0
    nodes = np.array(list(G.nodes()))
    for node_idx in range(len(nodes)):
        node = nodes[node_idx]
        # get the neighbours of a node
        neighbours = list(G.neighbors(node))
        temp_mAP = 0
        for neigh in neighbours:
            # define the circle's radius
            neigh_idx = np.where(nodes == neigh)[0]
            radius = distances[node_idx][neigh_idx][0]

            # find all the nodes within the circle
            radius_mask = distances[node_idx] <= radius
            # remove self loop
            radius_mask[node_idx] = False

            nodes_in_circle = nodes[radius_mask]
            # count how manyy should be there
            num = len(set(nodes_in_circle).intersection(set(neighbours)))
            # how many there are in total
            den = len(nodes_in_circle)

            temp_mAP  += num / den

        mAP += temp_mAP / len(neighbours) 
        
    mAP /= G.number_of_nodes()
    return mAP


def gencoordinates(m: int, n: int) -> Generator:
    seen = set()
    x, y = randint(m, n), randint(m, n)
    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(m, n), randint(m, n)
        while (x, y) in seen:
            x, y = randint(m, n), randint(m, n)


#def emd(tp0: Tuple, tp1: Tuple) -> Tuple:
#    g1, h1 = tp0
#    g2, h2 = tp1
#
#    mask1 = g1.final
#    mask2 = g2.final
#    
#    e1 = g1.pmu.data['e'][mask1]
#    e2 = g2.pmu.data['e'][mask2]
#    #e1 = g1.pmu.mass[mask1]
#    #e2 = g2.pmu.mass[mask2]
#
#    minimum = min(e1.sum(), e2.sum())
#    #energy_lost = abs(e1.sum() - e2.sum()) / minimum
#
#    reg = 0.005
#    reg_m_kl = 0.05
#    
#    m = cdist(h1[mask1], h2[mask2], metric=sqdist)
#    m /= np.max(m)
#    #M = ot.partial.partial_wasserstein(e1, e2, m, minimum)
#    M = ot.unbalanced.sinkhorn_unbalanced(e1, e2, m, reg, reg_m_kl)
#    #plt.imshow(M)
#    #plt.colorbar()
#    #plt.savefig('images/cost2.png')
#    #M = ot.unbalanced.sinkhorn_unbalanced(e1, e2, m, reg, reg_m_kl)
#
#    hyper_cost = np.sum(M * m)
#
#    '''
#    m = g1.pmu[mask1].delta_R(g2.pmu[mask2])
#    m /= np.max(m)
#    M = ot.partial.partial_wasserstein(e1, e2, m, minimum)
#    #M = ot.unbalanced.sinkhorn_unbalanced(e1, e2, m, reg, reg_m_kl)
#    
#    euclidean_cost = np.sum(M * m)
#    ''' 
#    return hyper_cost#, euclidean_cost, energy_lost
#    #return euclidean_cost

def emd(tp0: Tuple, tp1: Tuple) -> Tuple:
    g1, h1 = tp0
    g2, h2 = tp1

    mask1 = g1.final
    _mask1 = h1[:, 0] > 0
    mask2 = g2.final
    _mask2 = h2[:, 0] > 0 

    hyper_cost = 0
    for i in range(2):
        if i == 0:
            MASK1 = mask1 * _mask1
            MASK2 = mask2 * _mask2
        else:
            MASK1 = mask1 * ~_mask1
            MASK2 = mask2 * ~_mask2

        e1 = g1.pmu.data['e'][MASK1] 
        e2 = g2.pmu.data['e'][MASK2] 

        minimum = min(e1.sum(), e2.sum())
         
        m = cdist(h1[MASK1], h2[MASK2], metric=sqdist)
        m /= np.max(m)
        #M = ot.partial.partial_wasserstein(e1, e2, m, minimum)
        reg = 0.005
        reg_m_kl = 0.5
        hyper_cost += ot.unbalanced.sinkhorn_unbalanced2(e1, e2, m, reg, reg_m_kl)

        #hyper_cost += np.sum(M * m)
    return hyper_cost
     

def compute_emd(dataset, pair):
    data1 = dataset[pair[0]]
    data2 = dataset[pair[1]]

    keys = [k for k in data1.keys()]
    hyp, euc, energy = [], [], []
    for k in [3]:#range(1, 6, 2):
        g1, h1 = data1[keys[k]], data1[keys[k+1]]
        g2, h2 = data2[keys[k]], data2[keys[k+1]]
        #hyp_temp, euc_temp, energy_temp = emd(g1, g2, h1, h2)
        hyp.append(emd((g1,h1), (g2,h2)))
        #euc.append(euc_temp)
        #energy.append(energy_temp)

    return hyp#, euc, energy
