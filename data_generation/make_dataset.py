# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
from math import ceil
import copy

import click
from showerpipe.generator import PythiaGenerator
from showerpipe.lhe import split, LheData, count_events
from heparchy.write import HdfWriter
from heparchy.data.event import SignalVertex
from tqdm import tqdm

import sys
sys.path.append('/mainfs/home/gc2c20/myproject/hyperparticle/')
sys.path.append('/mainfs/home/gc2c20/myproject/hyperparticle/data_generation/')

import numpy as np
import graphicle as gcl
#from embedding import HyperEmbedding
from duplicates import duplicate_mask
from tree import FamilyTree

import networkx as nx
from hyperlib.embedding.sarkar import sarkar_embedding

@click.command()
@click.argument('lhe_path', type=click.Path(exists=True))
@click.argument('pythia_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(path_type=Path))
@click.argument('process_name',type=click.STRING)
def main(lhe_path, pythia_path, output_filepath,process_name):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    num_procs: int = comm.Get_size()

    total_num_events = count_events(lhe_path)
    stride = ceil(total_num_events / num_procs)

    # split filepaths for each process
    split_dir = output_filepath.parent
    split_fname = f'{output_filepath.stem}-{rank}{output_filepath.suffix}'
    split_path = split_dir / split_fname

    if rank == 0:  # split up the lhe file
        lhe_splits = split(lhe_path, stride)
        data = next(lhe_splits)
        for i in range(1, num_procs):
            comm.send(next(lhe_splits), dest=i, tag=10+i)
    else:
        data = comm.recv(source=0, tag=10+rank)

    gen = PythiaGenerator(pythia_path, data)
    if rank == 0:  # progress bar on root process
        gen = tqdm(gen)

    with HdfWriter(split_path) as hep_file:
        with hep_file.new_process(process_name) as proc:
            for event in gen:
                with proc.new_event() as event_write:
                    graph = gcl.Graphicle.from_numpy(
                        edges = event.edges,
                        pmu = event.pmu,
                        pdg = event.pdg,
                        status = event.status,
                        final = event.final,
                    )
                    #graph = duplicate_mask(g)

                    # cluster
                    algo = ['aKt', 'CA', 'Kt']
                    ps = [-1, 0, 1]
                    k=1
                    tree = FamilyTree(graph)
                    g = tree.history(p = ps[k])
                    if not g:
                        continue

                    event_write.set_pmu(graph.pmu.data)
                    event_write.set_pdg(graph.pdg.data)
                    event_write.set_status(graph.status.data)
                    event_write.set_edges(graph.edges)
                    event_write.set_mask('final', data=graph.final.data)
                                        
                    #tree = FamilyTree(graph)
                    for k in range(3):
                        lg = len(graph.pmu.data)
                        auxiliar_pmu = np.zeros_like(graph.pmu.data)
                        auxiliar_edges = np.zeros_like(graph.edges[:lg])
                        auxiliar_hyper = np.zeros((lg, 2))
                        auxiliar_mask = np.zeros(lg, dtype=bool)

                        g = tree.history(p = ps[k])
                        if not g:
                            continue
                        lab = np.where(g.nodes == -1)[0]

                        G = nx.Graph()
                        G.add_edges_from(g.edges)
                        nodes = np.array(G.nodes())
                        mapping = {nodes[i]: i for i in range(len(nodes))}
                        G = nx.relabel_nodes(G, mapping)
                        embed = sarkar_embedding(tree=G, root=mapping[-1], 
                            tau=0.7, weighted=False)
                        hyp = np.array(list(map(float, embed))).reshape(-1, 2)
                        
                        length = len(g.nodes)
                        auxiliar_pmu[:length] = g.pmu.data
                        auxiliar_edges[:length-1] = g.edges
                        auxiliar_hyper[:length] = hyp
                        auxiliar_mask[:length] = True

                        event_write.set_custom_dataset(
                            name = algo[k] + '_pmu', data = auxiliar_pmu, 
                            dtype = auxiliar_pmu.dtype
                        )
                        event_write.set_custom_dataset(
                            name = algo[k] + '_edges', data = auxiliar_edges, 
                            dtype = auxiliar_edges.dtype
                        )
                        event_write.set_custom_dataset(
                            name = algo[k] + '_hyp', data = auxiliar_hyper, 
                            dtype = auxiliar_hyper.dtype
                        ) 
                        event_write.set_custom_dataset(
                            name = algo[k] + '_mask', data = auxiliar_mask, 
                            dtype = auxiliar_mask.dtype
                        )


if __name__ == '__main__':
    sys.exit(main())
