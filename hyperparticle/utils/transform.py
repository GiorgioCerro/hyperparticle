import numpy as np
from numpy._typing import NDArray
from graphicle import Graphicle
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
    g.pmu.phi = translate(g.pmu.phi, jet_phi)
    return g

