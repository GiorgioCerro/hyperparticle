import sys
import os
from numpy._typing import NDArray

import click
from showerpipe.generator import PythiaGenerator, repeat_hadronize
from showerpipe.lhe import split, LheData, count_events

from tqdm import tqdm
import numpy as np
import graphicle as gcl
import matplotlib.pyplot as plt

from hyperparticle.tree import FamilyTree
import networkx as nx
from hyperlib.embedding.sarkar import sarkar_embedding
#from hyperparticle.utils.metrics import emd
from hyperparticle.utils.metrics import jet_angularities, eta_phi_dist
from hyperparticle.utils.transform import center

from sklearn.linear_model import LinearRegression
import ot

@click.command()
@click.argument('lhe_path', type=click.Path(exists=True))
@click.argument('pythia_path', type=click.Path(exists=True))
@click.argument('name', type=click.STRING)

def main(lhe_path, pythia_path, name):
    '''generate event and do analysis without writing any file
    '''
    #from mpi4py import MPI
    # 
    #comm = MPI.COMM_WORLD
    #rank: int = comm.Get_rank()
    #num_procs: int = comm.Get_size()

    #stride = 1
    #if rank==0:
    #    lhe_splits = split(lhe_path, stride)
    #    data = next(lhe_splits)
    #    for i in range(1, num_procs):
    #        comm.send(next(lhe_splits), dest=i, tag=10+i)
    #else: 
    #    data = comm.recv(source=0, tag=10+rank)

    data = next(split(lhe_path, 100))
    gen = PythiaGenerator(pythia_path, data)
    for i in range(10):
        g0 = gcl.Graphicle.from_event(next(gen))

        # hadronise the event
        hadronised_gen = repeat_hadronize(gen, 1)
        g1 = gcl.Graphicle.from_event(next(hadronised_gen))
        # cluster both the particle sets
        tree0 = FamilyTree(g0)
        tree1 = FamilyTree(g1)
        try: 
            g0 = tree0.history(R=1.0, p=-1, pt_cut=None, eta_cut=2.5, 
                recluster=None)
            g1 = tree1.history(R=1.0, p=-1, pt_cut=None, eta_cut=2.5, 
                recluster=None)
        except:
            print('Couldnt cluster')
            #break
            continue

        if g0 is None or g1 is None:
            print('Jet not found')
            continue

        
        # embed the jets and store hyperbolic coordinates
        hyps = []
        for g in [g0, g1]:
            lab = np.where(g.nodes == -1)[0]

            G = nx.Graph()
            G.add_edges_from(g.edges)
            nodes = np.array(G.nodes())
            mapping = {nodes[i]: i for i in range(len(nodes))}
            G = nx.relabel_nodes(G, mapping)

            embed = sarkar_embedding(tree=G, root=mapping[-1],
                    tau=0.6, weighted=False)
            hyp = np.array(list(map(float, embed))).reshape(-1, 2)
            hyps.append(hyp)


        g0 = center(g0)
        g1 = center(g1)
        print('event: ', i)
        print('jet ang: ', abs(jet_angularities(g0) - jet_angularities(g1)))
        print('energy diff: ',abs(max(g0.pmu.data['e']) - max(g1.pmu.data['e'])))


        distance_matrix = eta_phi_dist(g0, g1)
        energy = min(max(g0.pmu.data['e']), max(g1.pmu.data['e']))
        if i ==1:
            breakpoint()
        measure = ot.partial.partial_wasserstein2(
            a=g0.pmu.data['e'][g0.final], b=g1.pmu.data['e'][g1.final],
            M=distance_matrix, m=energy)
        #print('event: ', i)
        #print('jet ang: ', abs(jet_angularities(g0) - jet_angularities(g1)))
        print('emd: ', measure)
        #print('energy diff: ',abs(max(g0.pmu.data['e']) - max(g1.pmu.data['e'])))

        # center the jets - compute emd - compute jet angularities
        
        
    # send and recv 

    # create plot

if __name__ == '__main__':
    sys.exit(main())

