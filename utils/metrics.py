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


def precision_and_recall(dataset, n_event):
    '''Get precision and recall for all the different graphs.
    The average is over the n_event
    '''
    if n_event > dataset.__len__():
        raise Exception('Not enough events in the dataset')

    ##Here raise the error if n_event > len(dataset)
    precision = np.zeros((n_event, 4))
    recall = np.zeros((n_event, 4))

    event = dataset.__getitem__(0)
    keys = [k for k in event.keys()]
    for ev in range(n_event):
        event = dataset.__getitem__(ev)
        precision_temp = []
        recall_temp = []

        k = 0
        graph, hyp = event[keys[k]], event[keys[k+1]]
        hard_mask = hard_descendants(graph, 25)
        
        y_true = hard_mask[graph.final].astype(int)
        
        hyp_final = hyp[graph.final]
        dist = distance_matrix(hyp_final, off_diag=False)
        y_pred = np.array([y_true[np.argsort(j)[1]] for j in dist])

        precision_temp.append(precision_score(y_true, y_pred))
        recall_temp.append(recall_score(y_true, y_pred))

        for k in [2, 4, 6]:
            graph, hyp = event[keys[k]], event[keys[k+1]]
            hyp_final = hyp[graph.final, :]
            dist = distance_matrix(hyp_final, off_diag=False)
            y_pred = np.array([y_true[np.argsort(j)[1]] for j in dist])

            precision_temp.append(precision_score(y_true, y_pred))
            recall_temp.append(recall_score(y_true, y_pred))

        
        precision[ev, :] = precision_temp
        recall[ev, :] = recall_temp


    cols = ['MC', 'Anti-Kt', 'CA', 'Kt']
    rows = ['Precision', 'Recall']
    precision = np.mean(precision, axis=0)
    recall = np.mean(recall, axis=0)

    df = pd.DataFrame([precision, recall], index=rows, columns=cols)
    return df
