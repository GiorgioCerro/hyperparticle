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
from hyperparticle.utils.metrics import emd

from sklearn.linear_model import LinearRegression
@click.command()
@click.argument('lhe_path', type=click.Path(exists=True))
@click.argument('pythia_path', type=click.Path(exists=True))
@click.argument('name', type=click.STRING)
@click.argument('r_val', type=click.FLOAT)

def main(lhe_path, pythia_path, name, r_val):
    '''generate event and do analysis without writing any file
    '''
    from mpi4py import MPI
     
    comm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    num_procs: int = comm.Get_size()

    stride = 1
    if rank==0:
        lhe_splits = split(lhe_path, stride)
        data = next(lhe_splits)
        for i in range(1, num_procs):
            comm.send(next(lhe_splits), dest=i, tag=10+i)
    else: 
        data = comm.recv(source=0, tag=10+rank)


    gen = PythiaGenerator(pythia_path, data)
    for i in range(10):
        event0 = next(gen)
        g0 = gcl.Graphicle.from_numpy(
            edges = event0.edges,
            pmu = event0.pmu,
            pdg = event0.pdg,
            status = event0.status,
            final = event0.final,
        )

        # hadronise the event
        hadronised_gen = repeat_hadronize(gen, 1)
        event1 = next(hadronised_gen)
        g1 = gcl.Graphicle.from_numpy(
            edges = event1.edges,
            pmu = event1.pmu,
            pdg = event1.pdg,
            status = event1.status,
            final = event1.final,
        )
 
        # cluster both the particle sets
        tree0 = FamilyTree(g0)
        tree1 = FamilyTree(g1)

        try: 
            g0 = tree0.history(R=r_val, p=-1, pt_cut=None, eta_cut=2.5, 
                recluster=None)
            g1 = tree1.history(R=r_val, p=-1, pt_cut=None, eta_cut=2.5, 
                recluster=None)
        except:
            break

        if g0 is None or g1 is None:
            break

        
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

        
        # compute the EMD and the jet angularities


    # send and recv 

    # create plot

if __name__ == '__main__':
    sys.exit(main())

