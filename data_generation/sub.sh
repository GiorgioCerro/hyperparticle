#!/bin/bash

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=0:40:00          # walltime

# mail alert at start, end and abortion
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=g.cerro@soton.ac.uk

file_in="../data/compare_algo/hz.lhe.gz"
file_out="../data/compare_algo/hz_train.hdf5"

source activate pyg
mpiexec -n 40 python3 make_dataset.py $file_in pythia-settings.cmnd $file_out signal
source deactivate
