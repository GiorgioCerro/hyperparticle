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
sys.path.append('/mainfs/home/gc2c20/myproject/hyperTree/')
sys.path.append('/mainfs/home/gc2c20/myproject/hyperTree/scripts/ \
        data_generation/')

import numpy as np
import graphicle as gcl
from embedding import HyperEmbedding
from duplicates import duplicate_mask

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
                    #event_write.set_pmu(event.pmu)
                    #event_write.set_pdg(event.pdg)
                    #event_write.set_status(event.status)
                    #event_write.set_color(event.color)
                    #event_write.set_helicity(event.helicity)
                    #event_write.set_edges(event.edges)
                    #event_write.set_mask('final', data=event.final)

                    g = gcl.Graphicle.from_numpy(
                        edges = event.edges,
                        pmu = event.pmu,
                        pdg = event.pdg,
                        status = event.status,
                        final = event.final,
                    )
                    graph = duplicate_mask(g)
                    graph_node = copy.deepcopy(graph)
                    graph_node.adj = gcl.transform.particle_as_node(
                        graph_node.adj)

                    hyper_coords = np.NaN
                    while np.isnan(hyper_coords).sum() > 0:
                        hyper = HyperEmbedding(graph_node)
                        hyper.get_embedding()
                        hyper_coords = hyper.embeddings

                
                    event_write.set_pmu(graph.pmu.data)
                    event_write.set_pdg(graph.pdg.data)
                    event_write.set_status(graph.status.data)
                    event_write.set_edges(graph.edges)
                    event_write.set_mask('final', data=graph.final.data)

                    event_write.set_custom_dataset(
                        name='hyper_coords',data=hyper_coords,
                        dtype='double')
                    
if __name__ == '__main__':
    sys.exit(main())
