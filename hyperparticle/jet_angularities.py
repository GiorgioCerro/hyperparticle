import sys
import os
from numpy._typing import NDArray

import click
from showerpipe.generator import PythiaGenerator, repeat_hadronize
from showerpipe.lhe import split, LheData
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import graphicle as gcl
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist

from hyperparticle.tree import FamilyTree
#from hyperparticle.utils.metrics import emd
from hyperparticle.utils.metrics import jet_angularities, eta_phi_dist
from hyperparticle.utils.transform import center, get_weights, get_embedding

from scipy import stats
import ot

@click.command()
@click.argument('lhe_path', type=click.Path(exists=True))
@click.argument('pythia_path', type=click.Path(exists=True))
@click.argument('space', type=click.STRING)
@click.argument('name', type=click.STRING)

def main(lhe_path, pythia_path, space, name):
    '''generate event and do analysis without writing any file
    '''
    from mpi4py import MPI
     
    comm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    num_procs: int = comm.Get_size()

    stride = 500
    if rank==0:
        lhe_splits = split(lhe_path, stride)
        data = next(lhe_splits)
        for i in range(1, num_procs):
            comm.send(next(lhe_splits), dest=i, tag=10+i)
    else: 
        data = comm.recv(source=0, tag=10+rank)

    #data = next(split(lhe_path, 10))
    gen = PythiaGenerator(pythia_path, data)
    total_emd = []
    total_ang = []
    pts = []

    for event in tqdm(gen):
        g0 = gcl.Graphicle.from_event(event)
        # hadronise the event
        hadronised_gen = repeat_hadronize(gen, 1)
        try:
            g1 = gcl.Graphicle.from_event(next(hadronised_gen))
        except:
            # this happens if the iterator is empty because pythia cannot hadronise
            continue
        # cluster both the particle sets
        tree0 = FamilyTree(g0)
        tree1 = FamilyTree(g1)
        # change this shit§
        try: 
            _g0 = tree0.history(R=1.0, p=-1, pt_cut=30, eta_cut=2.5, 
                recluster=None)
            _g1 = tree1.history(R=1.0, p=-1, pt_cut=30, eta_cut=2.5, 
                recluster=None)
        except:
            #print('Couldnt cluster')
            #break
            continue

        if _g0 is None or _g1 is None or len(_g0) != len(_g1):
            #print('Jet not found')
            continue
       
        for k in range(len(_g0)):
            g0 = _g0[k]
            g1 = _g1[k]
            if max(g0.pmu.pt) > 550 or max(g0.pmu.pt) < 500:
                continue
            if max(g1.pmu.pt) > 550 or max(g1.pmu.pt) < 500:
                continue

            # embed the jets and store hyperbolic coordinates
            hyps = []
            for g in [g0, g1]:
                # assign weights
                g = get_weights(g)
                
                #get the embeddings
                hyp = get_embedding(g)
                hyps.append(hyp)


            g0 = center(g0)
            g1 = center(g1)
            if space == "euclidean":
                distance_matrix = eta_phi_dist(g0, g1)
            if space == "hyperbolic":
                distance_matrix = cdist(hyps[0][g0.final], hyps[1][g1.final],
                    metric=hdist)
                distance_matrix /= np.max(distance_matrix)
            energy = min(max(g0.pmu.pt), max(g1.pmu.pt)) - 1e-8
            _a = g0.pmu.pt[g0.final]
            _b = g1.pmu.pt[g1.final]
            #_a /= sum(_a)
            #_b /= sum(_b)
            try:
                emd_measure = ot.partial.partial_wasserstein2(
                    a=_a, b=_b, M=distance_matrix, m=energy)
                #emd_measure = ot.emd2(a=_a, b=_b, M=distance_matrix)
            except:
                breakpoint()
            angular_measure = abs(jet_angularities(g0) - jet_angularities(g1))
            total_emd.append(emd_measure) 
            total_ang.append(angular_measure)
        
            pts.append(max(g0.pmu.pt))
            pts.append(max(g1.pmu.pt))

            #if emd_measure < angular_measure:
            #    breakpoint()
        
    if rank!=0:
        comm.send(total_emd, dest=0, tag=0)
        comm.send(total_ang, dest=0, tag=1)
        comm.send(pts, dest=0, tag=2)
    else: 
        total_emd = [total_emd]
        total_ang = [total_ang]
        pts = [pts]
        for i in range(1, num_procs):
            total_emd.append(comm.recv(source=i, tag=0))
            total_ang.append(comm.recv(source=i, tag=1))
            pts.append(comm.recv(source=i, tag=2))

        EMD = np.array([j for item in total_emd for j in item])
        angular = np.array([j for item in total_ang for j in item])
        pt = np.array([j for item in pts for j in item])
        #print(np.mean(EMD), np.std(EMD), np.max(EMD))
        plot(EMD, angular, f'images/{name}')


def plot(_x, _y, name_path):
    side = 40
    plt.plot(range(side), range(side), '--', c='r', label='bound')
    plt.fill_between(range(side), range(side), side*[side-1], color='pink')
    #plt.scatter(_x, _y, label='data points')

    
    # density estimation
    X, Y = np.mgrid[0:side:40j, 0:side:40j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([_x, _y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    plt.imshow(np.rot90(Z), cmap='Greys', extent=[0,side,0,side])
    
    plt.colorbar()
    plt.legend()
    plt.title(f"EMD: QCD Jet Angularity ({len(_x)} datapoints)")
    plt.xlim(0, side-1)
    plt.ylim(0, side-1)
    plt.xlabel("Parton-Hadron EMD (GeV)")
    plt.ylabel("Angularity Modification (GeV)")

    plt.savefig(name_path)
    plt.close()


def hdist(x, y):
    sq_norm_x = np.linalg.norm(x, axis=-1) ** 2.
    sq_norm_y = np.linalg.norm(y, axis=-1) ** 2.
    sq_norm_xy = np.linalg.norm(x - y, axis=-1) ** 2.

    denom = np.clip(a =((1 - sq_norm_x) * (1 - sq_norm_y)), a_min=1e-8, a_max=None)
    cosh_angle = 1 + 2 * sq_norm_xy / denom
    cosh_angle = np.clip(a=cosh_angle, a_min=1. + 1e-8, a_max=None)
    dist = np.arccosh(cosh_angle)
    return dist ** 2.


if __name__ == '__main__':
    sys.exit(main())

