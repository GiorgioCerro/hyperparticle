#!/bin/bash

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=0:45:00          # walltime

# mail alert at start, end and abortion
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=g.cerro@soton.ac.uk

source activate pyg
mpiexec -n 40 python3 make_dataset.py ../data/test3/hz.lhe.gz pythia-settings.cmnd ../data/test3/hz_train.hdf5 signal
source deactivate
