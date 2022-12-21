#!/bin/bash

#SBATCH --ntasks-per-node=20     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=0:30:00          # walltime

# mail alert at start, end and abortion
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=g.cerro@soton.ac.uk

file_in="../data/repeat_hadronise/hz.lhe.gz"
file_out="../data/repeat_hadronise/hadron.hdf5"

source activate pyg
mpiexec -n 20 python3 make_hadronise.py $file_in p2.cmnd $file_out signal
source deactivate
