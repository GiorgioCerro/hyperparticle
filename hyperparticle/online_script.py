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


    if name == 'hadron':
        n_events = 200
        first_gen = PythiaGenerator(pythia_path, data)
        next(first_gen)
        gen = repeat_hadronize(first_gen, n_events)

    else: 
        n_events = 1000
        data = LheData(data).repeat(n_events)
        gen = PythiaGenerator(pythia_path, data)

    rg = range(int(n_events/2))
    if rank==0:
       rg = tqdm(rg) 

    energy = []
    mass = []
    pt = []
    hyper = []
    delta1 = []
    delta2 = []
    for i in rg:
        event_dict = {}
        two_branches = True
        for k in range(2):
            event = next(gen)
            graph = gcl.Graphicle.from_numpy(
                    edges = event.edges,
                    pmu = event.pmu, 
                    pdg = event.pdg, 
                    status = event.status,
                    final = event.final,
            )

            tree = FamilyTree(graph)
            try: 
                g = tree.history(R=r_val, p=-1, pt_cut=None, 
                            eta_cut=2.5, recluster=None)
            except:
                break

            if g is None:
                break

            lab = np.where(g.nodes == -1)[0]

            G = nx.Graph()
            G.add_edges_from(g.edges)
            nodes = np.array(G.nodes())
            mapping = {nodes[i]: i for i in range(len(nodes))}
            G = nx.relabel_nodes(G, mapping)

            embed = sarkar_embedding(tree=G, root=mapping[-1],
                    tau=0.6, weighted=False)
            hyp = np.array(list(map(float, embed))).reshape(-1, 2)

            if np.sum(hyp[:, 0][g.final] > 0) < 4 or np.sum(hyp[:, 0][g.final] < 0) < 4:
                two_branches = False #not enough particles in the two jets

            #algo = ['aKt', 'aKt+CA']
            event_dict[f'event_{k}'] = (g, hyp)

        if len(event_dict) < 2 or two_branches == False:
            continue

        #hyp_aKt.append(
        #        emd(event_dict['event_0_aKt'], event_dict['event_1_aKt']))
        hyper.append(
                emd(event_dict['event_0'], event_dict['event_1']))
        energy.append(np.mean([
                max(event_dict['event_0'][0].pmu.data['e']),
                max(event_dict['event_1'][0].pmu.data['e'])]))
        pt.append([
                (max(event_dict['event_0'][0].pmu.pt),
                max(event_dict['event_1'][0].pmu.pt))])
        mass.append(np.mean([
                max(event_dict['event_0'][0].pmu.mass),
                max(event_dict['event_1'][0].pmu.mass)]))
    '''
    hard_mask = graph.hard_mask['outgoing'].data
    b_idx = np.where(abs(graph.pdg[hard_mask].data) == 5)[0]
    pmu1 = graph.pmu[hard_mask][b_idx[0]]
    pmu2 = graph.pmu[hard_mask][b_idx[1]]
    delta = delta_R(pmu1, pmu2)
    '''
    if len(hyper) == 0:
        hyper = np.zeros(5)
        energy = 1000*np.ones(5)
        mass = 1000*np.ones(5)
        pt = 1000*np.ones(5)
    hyper_std = np.std(hyper)
    hyper = np.nanmean(hyper)
    energy = np.mean(energy)
    mass = np.mean(mass)
    pt = np.mean(pt)
    

    
    if rank != 0:
        comm.send(hyper, dest=0, tag=0)
        comm.send(energy, dest=0, tag=1)
        comm.send(pt, dest=0, tag=2)
        comm.send(mass, dest=0, tag=3)
        comm.send(hyper_std, dest=0, tag=4)

    if rank == 0:
        hyp0 = [hyper]
        _energy = [energy]
        _mass = [mass]
        _pt = [pt]
        _std = [hyper_std]

        for i in range(1, num_procs):
            hyp0.append(comm.recv(source=i, tag=0))
            _energy.append(comm.recv(source=i, tag=1))
            _pt.append(comm.recv(source=i, tag=2))
            _mass.append(comm.recv(source=i, tag=3))
            _std.append(comm.recv(source=i, tag=4))

        save_plot(hyp0, _std, _energy, 'images/'+str(r_val)[::2]+'R_'+name+'.png')
    
        print('Done')


def save_plot(data, _std, _x, path):
        _x = np.array(_x)
        data = np.array(data)
        _std = np.array(_std)
        mask = _x < 550
        _x = _x[mask]
        labels = ['aKt']

        mask2 = data[mask] < 20
        Y = data[mask][mask2] #- 0.25*_x[mask2] 
        Yerr = _std[mask][mask2]
        plt.errorbar(_x[mask2], Y, yerr=Yerr, fmt='o', label=labels[0])

        reg = LinearRegression().fit(_x[mask2][:, None], Y)
        b = reg.intercept_
        m = reg.coef_[0]
        plt.plot(_x[mask2], m*_x[mask2] + b, 
                label=f'{labels[0]} - slope: {m:.2f}, a0: {b:.2f}')
        
        plt.xlabel('energy')
        plt.ylabel('EMD')
        plt.legend()
        plt.xlim(0,600)
        #plt.ylim(0, 50)
        plt.title(path)
        plt.savefig(path)
        plt.close()

def delta_R(pmu1, pmu2):
    _xy_pol1 = pmu1.data['x'] + 1.0j * pmu1.data['y']
    _xy_pol2 = pmu2.data['x'] + 1.0j * pmu2.data['y']
    dphi = np.angle(_xy_pol1 * _xy_pol2)

    deta = pmu1.eta - pmu2.eta
    return np.hypot(deta, dphi)

if __name__ == '__main__':
    sys.exit(main())

