import numpy as np
from numpy._typing import NDArray
from graphicle import Graphicle
import networkx as nx
from hyperlib.embedding.sarkar import sarkar_embedding 
from copy import deepcopy

def rotate(phi: NDArray, dphi: float) -> NDArray:
    """Rotate event along phi axis
    """
    rot_pos = np.exp(1.0j * dphi).conjugate()
    phi = np.exp(1.0j * phi)
    return np.angle(phi * rot_pos)


def translate(eta: NDArray, deta: float) -> NDArray:
    """Translate event along eta axis
    """
    return eta - deta


def center(graph: Graphicle) -> Graphicle:
    """Pass a graphicle object to center the jet axis
    """
    g = deepcopy(graph)
    jet_id = np.argmax(g.pmu.pt)
    jet_eta, jet_phi = g.pmu.eta[jet_id], g.pmu.phi[jet_id]
    
    g.pmu.eta = translate(g.pmu.eta, jet_eta)
    g.pmu.phi = rotate(g.pmu.phi, jet_phi)
    return g


def get_weights(graph: Graphicle) -> Graphicle:
    """Return a pt weighted graph
    """
    n_dict = {graph.nodes[i]: i for i in range(len(graph.nodes))}
    weights = []
    pts = graph.pmu.pt
    for i in range(0, len(graph.edges), 2):
        a, b = graph.edges[i][1], graph.edges[i+1][1]
        pt1, pt2 = pts[n_dict[a]], pts[n_dict[b]]
        weights.append(pt1 / (pt1+pt2))
        weights.append(pt2 / (pt1+pt2))

    graph.adj.weights = weights
    return graph


def get_embedding(graph: Graphicle, tau: float=0.6, 
                    weighted: bool=True) -> NDArray:
    """Embed tree with the sarkar algorithm
    """
    #lab = np.where(g.nodes == -1)[0]
    G = nx.Graph()
    if weighted:
        weighted_edges = [
        (graph.edges[i][0], graph.edges[i][1], graph.adj.weights[i]) 
        for i in range(len(graph.edges))
        ]
        G.add_weighted_edges_from(weighted_edges)
    else:
        G.add_edges_from(graph.edges)

    nodes = np.array(G.nodes())
    mapping = {nodes[i]: i for i in range(len(nodes))}
    G = nx.relabel_nodes(G, mapping)

    embed = sarkar_embedding(tree=G, root=mapping[-1], tau=tau, 
                            weighted=weighted)
    _hyp = np.array(list(map(float, embed))).reshape(-1, 2)
    hyp = np.array([_hyp[mapping[node]] for node in graph.nodes])
    return hyp
