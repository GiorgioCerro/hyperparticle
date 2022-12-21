import numpy as np
import networkx as nx
from .visualisation import hard_descendants
from sklearn.metrics import precision_score, recall_score
import pandas as pd


def sqdist(x, y):
    sq_norm_x = np.linalg.norm(x, axis=-1) ** 2.
    sq_norm_y = np.linalg.norm(y, axis=-1) ** 2.
    sq_norm_xy = np.linalg.norm(x - y, axis=-1) ** 2.

    cosh_angle = 1 + 2 * sq_norm_xy / ((1 - sq_norm_x) * (1 - sq_norm_y))
    cosh_angle = np.clip(a=cosh_angle, a_min=1. + 1e-8, a_max=None)
    dist = np.arccosh(cosh_angle)
    return dist ** 2.


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


def mAP(event):
    '''Get the mean average precision for all the different graphs.
    '''
    keys = [k for k in event.keys()]
    total_mAP = []

    for k in range(0, 7, 2):
        graph, hyp = event[keys[k]], event[keys[k+1]]
        G = nx.Graph()
        G.add_edges_from(graph.edges)

        distances = distance_matrix(hyp, off_diag=False)
        mAP = 0
        for node_idx in range(len(graph.nodes)):
            node = graph.nodes[node_idx]
            # get the neighbours of a node
            neighbours = list(G.neighbors(node))
            temp_mAP = 0
            for neigh in neighbours:
                # define the circle's radius
                neigh_idx = np.where(graph.nodes == neigh)[0]
                radius = distances[node_idx][neigh_idx][0]

                # find all the nodes within the circle
                radius_mask = distances[node_idx] <= radius
                # remove self loop
                radius_mask[node_idx] = False

                nodes_in_circle = graph.nodes[radius_mask]
                # count how manyy should be there
                num = len(set(nodes_in_circle).intersection(set(neighbours)))
                # how many there are in total
                den = len(nodes_in_circle)

                temp_mAP  += num / den

            mAP += temp_mAP / len(neighbours) 
            
        mAP /= G.number_of_nodes()
        total_mAP.append(mAP)

    
    return total_mAP


