#!/bin/bash

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=1:00:00          # walltime

# mail alert at start, end and abortion
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=g.cerro@soton.ac.uk

file_in_1="../data/jet_tagging/train_sig/hz.lhe.gz"
file_out_1="../data/jet_tagging/train_sig/signal.hdf5"

file_in_2="../data/jet_tagging/train_bkg/gz.lhe.gz"
file_out_2="../data/jet_tagging/train_bkg/background.hdf5"


source activate pyg
mpiexec -n 40 python3 make_dataset.py $file_in_1 pythia-settings.cmnd $file_out_1 signal
mpiexec -n 40 python3 make_dataset.py $file_in_2 pythia-settings.cmnd $file_out_2 background
source deactivate
