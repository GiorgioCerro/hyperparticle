#!/bin/bash

#SBATCH --ntasks-per-node=5     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=0:10:00          # walltime

# mail alert at start, end and abortion
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=g.cerro@soton.ac.uk

source activate pyg
mpiexec -n 5 python3 make_dataset.py ../data/higgs_tagging/signal/hz.lhe.gz pythia-settings.cmnd ../data/higgs_tagging/signal/hz_train.hdf5 signal
source deactivate
