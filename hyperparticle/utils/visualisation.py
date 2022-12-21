'''Visualization utils.'''
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy
import random

def mobius_add(x, y):
    '''Mobius addition in numpy'''
    xy = np.sum(x * y, 1, keepdims=True)
    x2 = np.sum(x * x, 1, keepdims=True)
    y2 = np.sum(y * y, 1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den


def mobius_mul(x, t):
    '''Mobius multiplication in numpy.'''
    normx = np.sqrt(np.sum(x * x, 1, keepdims=True)) + 1e-12
    return np.tanh(t * np.arctanh(normx)) * x / normx


def geodesic_fn(x, y, nb_points=100):
    '''Get coordinates of points on the geodesic between x and y.'''
    t = np.linspace(0, 1, nb_points)
    x_rep = np.repeat(x.reshape((1, -1)), len(t), 0)
    y_rep = np.repeat(y.reshape((1, -1)), len(t), 0)
    t1 = mobius_add(-x_rep, y_rep)
    t2 = mobius_mul(t1, t.reshape((-1, 1)))
    return mobius_add(x_rep, t2)


def plot_geodesic(x, y, ax):
    '''Plots geodesic between x and y.'''
    points = geodesic_fn(x, y)
    ax.plot(points[:, 0], points[:, 1], color='black', linewidth=1., alpha=1)


def get_labels(graph):
    edg = np.where(graph.edges['out'] == -1)[0]
    nodes = graph.edges['in'][edg]
    idx = [np.where(graph.nodes == n)[0][0] for n in nodes]
        
    # Create the networkx object
    G = nx.DiGraph()
    #ed = copy.copy(graph.edges)
    #ed['in'] = graph.edges['out']
    #ed['out'] = graph.edges['in']
    G.add_edges_from(graph.edges)
        
    labels = np.zeros(len(graph.nodes), dtype=int)
    for j in nodes:
        descendants = list(nx.descendants(G, j))
        descendants.append(j)
        idxs = [np.where(graph.nodes == n)[0][0] for n in descendants]
        if len(idxs) > 1:
            labels[idxs] = max(labels)+1
        else:
        # in this way pseudojets with no children are all of the same colours
            labels[idxs] = 0
            labels[np.where(graph.nodes == -1)[0][0]] = 0

    return labels


def get_colors(graph, color_seed=1234):
    """random color assignment for label classes."""
    y = get_labels(graph)
        
    np.random.seed(color_seed)
    colors = {}
    number_of_colors = len(np.unique(y))
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') 
        for j in range(6)])
            for k in range(number_of_colors)]
    #for k in np.unique(y):
    #    r = np.random.random()
    #    b = np.random.random()
    #    g = np.random.random()
    #    colors[k] = (r, g, b)
    return np.array([colors[k] for k in y])


def hard_descendants(graph, idx):
    node = graph.nodes[np.where(graph.pdg.data == idx)[0][0]]

    #Create the nx obj
    G = nx.DiGraph()
    G.add_edges_from(graph.edges)

    descendants = list(nx.descendants(G, node))
    descendants.append(node)
    
    idxs = np.array([np.where(graph.nodes == j)[0] for j in descendants])
    mask = np.zeros_like(graph.nodes, dtype=bool)
    mask[idxs] = True
    return mask
