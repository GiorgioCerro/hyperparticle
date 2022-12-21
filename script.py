import glob
import sys
import numpy as np
from numpy._typing import NDArray
import matplotlib.pyplot as plt
from utils.metrics import compute_emd, gencoordinates
from data_handler import OneSet
from tqdm import tqdm
import click

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.argument('name', type=click.STRING)

def main(path, name):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    #path = 'data/repeat_shower/'
    files = glob.glob(path + '/*.hdf5')
    dataset = OneSet(files[rank])

    gen = gencoordinates(0, len(dataset)-1)
    hyp, euc, energy = [],[],[]

    rg = range(1500)
    if rank==0: 
        rg=tqdm(rg)

    for i in rg:
        pair = next(gen)
        #hyp_temp, euc_temp, energy_temp = compute_emd(dataset, pair)

        hyp.append(compute_emd(dataset, pair))
        #euc.append(euc_temp)
        #energy.append(energy_temp)


  
    hyp_mean = list(np.mean(hyp, axis=0))
    hyp_std = list(np.std(hyp, axis=0))
    #euc_mean = list(np.mean(euc, axis=0))
    #euc_std = list(np.std(euc, axis=0))
    #e_mean = list(np.mean(energy, axis=0))
    #e_std = list(np.mean(energy, axis=0))

    if rank != 0:
        comm.send(hyp_mean, dest=0, tag=0)
        comm.send(hyp_std, dest=0, tag=1)
        #comm.send(euc_mean, dest=0, tag=2)
        #comm.send(euc_std, dest=0, tag=3)
        #comm.send(e_mean, dest=0, tag=4)
        #comm.send(e_std, dest=0, tag=5)

    if rank == 0:
        hyperbolic_mean = [hyp_mean]
        hyperbolic_std = [hyp_std]
        #euclidean_mean = [euc_mean]
        #euclidean_std = [euc_std]
        #energy_mean = [e_mean]
        #energy_std = [e_std]
        for i in range(1, num_procs):
            hyperbolic_mean.append(comm.recv(source=i, tag=0))
            hyperbolic_std.append(comm.recv(source=i, tag=1))
            #euclidean_mean.append(comm.recv(source=i, tag=2))
            #euclidean_std.append(comm.recv(source=i, tag=3))
            #energy_mean.append(comm.recv(source=i, tag=4))
            #energy_std.append(comm.recv(source=i, tag=5))

        h_mean = np.array(hyperbolic_mean)
        h_std = np.array(hyperbolic_std)
        #e_mean = np.array(euclidean_mean)
        #e_std = np.array(euclidean_std)
        #energy_mean = np.array(energy_mean)
        #energy_std = np.array(energy_std)
        #saveresult((h_mean, h_std), (e_mean, e_std), 'images/'+name+'.png')
        #plot_energy(h_mean, h_std, energy_mean, energy_std, 
        #        'images/h_'+name+'.png')
        #plot_energy(e_mean, e_std, energy_mean, energy_std, 
        #        'images/e_'+name+'.png')
        save_one(h_mean, 'images/akt_'+name+'.png')

        print('Done!')

def saveresult(data1: tuple, data2: tuple, path: str) -> None:
    h_mean, h_std = data1
    e_mean, e_std = data2

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.flatten()

    titles = ['EMD in Hyperbolic space', 'EMD in Eta-Phi space']
    labels = ['aKt', 'CA', 'Kt']
    col = ['#1f77b4', '#ff7f0e', '#2ca02c']
    length = np.array(range(len(h_mean)))
    for i in range(3):
        #ax[0].scatter(length+0.1*i, h_mean[:, i], c=col[i], label=labels[i])
        #ax[1].scatter(length+0.1*i, e_mean[:, i], c=col[i], label=labels[i])

        ax[0].errorbar(length+0.1*i, h_mean[:, i], yerr=h_std[:, i], 
                c=col[i], fmt='o', label=labels[i])
        ax[1].errorbar(length+0.1*i, e_mean[:, i], yerr=e_std[:, i], 
                c=col[i], fmt='o')
    
    for j in length:
        for i in range(3):
            ax[2].scatter(j, h_mean[j, i] / max(h_mean[j]+1e-8), c=col[i])
            ax[3].scatter(j, e_mean[j, i] / max(e_mean[j]+1e-8), c=col[i])

    titles = ['EMD in Hyperbolic space', 'EMD in Eta-Phi space']
    for i in range(2):
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Event number')
        ax[i].set_ylabel('EMD')

    ax[0].legend()
    plt.savefig(path)
    plt.close()


def save_one(data: NDArray, path: str) -> None:
    '''save only one specific algorithm
    '''
    length = np.array(range(len(data)))
    plt.scatter(length, data)
    plt.ylim(0, 1000)
    plt.savefig(path)
    plt.close()


def plot_energy(data, data_std, energy, energy_std, path):
    labels = ['aKt', 'CA', 'Kt']
    for i in range(3):
        plt.scatter(energy[:, i], data[:, i], label=labels[i])
        plt.errorbar(energy[:, i], data[:, i],
                xerr=energy_std[:, i], yerr=data_std[:, i], fmt='o')
    plt.legend()
    plt.savefig(path)
    plt.close()


if __name__=='__main__':
    sys.exit(main())
