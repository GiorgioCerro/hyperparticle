# hyperparticle

This project aims to embed particles showers into hyperbolic spaces.

# data4K branch

This branch is for my collegue who needs only the particles' trees.
This is a guide for how to use the data.

# data

The data files are hdf5.

A lhe file is provided in the data folder. To generate the data go to the 
data geneartion folder and run:

`sbatch sub.sh` 

Before doing so, make sure to check the settings in the pythia setting file 
and in the sbatch file.

# data_handler
    
Use the data handler in order to get the dataloader. For each event you will 
have a dictionary with four different graphs: MC_graph, Anti-Kt_graph, 
CA_graph, Kt_graph, which correspond to the MonteCarlo truth and the three
clustering algorithms respectively. 

See the 'Tutorial.ipynb' for some example.

# conda environment

To install all the necessary packages, it is possible to create a new 
environment using the yml file. To do so run the following command:

`conda env create -f environment.yml`

