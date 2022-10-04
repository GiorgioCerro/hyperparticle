import numpy as np
from graphicle import Graphicle
import graphicle as gcl
from numpy import array
from pyjet import cluster
from pyjet import PseudoJet
from numpy.lib.recfunctions import append_fields
from typing import Tuple


class FamilyTree:
    def __init__(self, graph: Graphicle):
        self.graph = graph
        #self.event = self.__getevent__()
        self.pyjet_dtype = {
            'names': ('E', 'px', 'py', 'pz'), 
            'formats': ('f8', 'f8', 'f8', 'f8')
        }
        self.gcl_dtype = {
            'names': ('x', 'y', 'z', 'e', 'id'),
            'formats': ('f8', 'f8', 'f8', 'f8', 'f8')
        }
        self.event = self.__getevent__()

    
    def __getevent__(self) -> array:
        '''Get the final state particles and put them in the correct format
        for pyjet.

        Returns
        -------
        event : array-like
            final state particles' four momentum
        '''
        mask = self.graph.final.data
        event = self.graph.pmu.data[mask][
            ['e', 'x', 'y', 'z']
        ].copy().astype(self.pyjet_dtype)

        event = append_fields(event, 'id', data=np.arange(len(event)))
        #event = append_fields(event, 'id', data=self.graph.nodes[mask])
        return event
    

    def __get1tree__(self, jet: PseudoJet, first_id: int) -> Tuple[list,list]:
        '''Given a PseudoJet object, reconstruct the tree structure, 
        returning a list of all the particles and a list of edges.
        Leaves have positive label; ancestor's label depends on others jets.

        Parameters
        ----------
        jet : PseudoJet
            The jet to reconstruct
        first_id : int
            Negative label to start with

        Returns
        -------
        ptcl : list
            List of all the particles in the jet
        edges : list
            List of all the edges of the graph
        '''
        ptcl = []
        #ptcl.append((jet.pt, jet.eta, jet.phi, jet.mass, first_id))

        # IMPORTANT: This edge helps to create a fully connected graph
        #edges = [(first_id, -1)]
        if not jet.parents:
            ptcl.append((jet.px, jet.py, jet.pz, jet.e, jet.id))
            edges = [(-1, jet.id)]
            return ptcl, edges
        ptcl.append((jet.px, jet.py, jet.pz, jet.e, first_id))
        edges = [(-1, first_id)]

        children = jet.parents
        parents_id = [first_id, first_id]
        children_temp = [0]

        _id = first_id - 1
        while len(children_temp) > 0:
            children_temp = []
            parents_id_temp = []
            for n, d in enumerate(children):
                if d.parents:
                    # if a particle has a children collect them
                    children_temp.append(d.parents[0])
                    children_temp.append(d.parents[1])
                    # collect the particle
                    ptcl.append((d.px, d.py, d.pz, d.e, _id))
                    # collect the id
                    parents_id_temp.append(_id)
                    parents_id_temp.append(_id)
                    # collect the edge of the particle and its parent
                    edges.append((parents_id[n], _id))
                    # update id
                    _id += -1
                
                else:
                    # if a particle doesn't have children collect the ptcl
                    ptcl.append((d.px, d.py, d.pz, d.e, d.id))
                    # collect the edge
                    edges.append((parents_id[n], d.id))
            # update list of children and ids
            children = children_temp
            parents_id = parents_id_temp

        return ptcl, edges
            

    def __tographicle__(self, particles: array, edges: array) -> Graphicle:
        '''Create a Graphicle object from a list of particles and a list of 
        edges between them.

        Parameters
        ----------
        particles: array
            particles of the event
        edges: array
            list of edges 

        graph: Graphicle
            Graphicle object with momentum and edges
        '''
        momentum = particles[:, :4]
        indices = particles[:, 4].astype(int)
        graph = gcl.Graphicle.from_numpy(
            edges = edges,
            final = np.ones(len(particles), dtype=bool),
            pdg = np.ones(len(particles)),
        )

        order = [np.where(indices == idx)[0][0] for idx in graph.nodes]
        graph.pmu.data = momentum[order]
        
        finals = graph.nodes >= 0
        graph.final.data = finals
        return graph

        
    def history(self, R=0.8, p=1) -> Tuple[array, array]:
        '''Custer the event and return a tree graph with all the
        jets connected to a common ancestor with label -1 and p=(0,0,0,0)
        
        Parameters
        ----------
        jets : list
            List of PseudoJet objects

        Returns
        -------
        ptcls : array
            all the particles in the graph and the ancestor with zero momentum
        edges : array
            all the edges between particles
        '''
        sequence = cluster(self.event, R=R, p=p, ep=True)
        jets = sequence.inclusive_jets()

        ptcl = [[(0., 0., 0., 0., -1)]]
        edge = []
        first_id = -2
        for jet in jets:
            ptcl_temp, edge_temp = self.__get1tree__(jet, first_id)
            ptcl.append(ptcl_temp)
            first_id = min(first_id, np.min(edge_temp)) - 1
            edge.append(edge_temp)
        
        edges = np.array([item for sublist in edge for item in sublist])
        ptcls = np.array([item for sublist in ptcl for item in sublist])
        #ptcls = ptcls.astype(self.gcl_dtype)
        #ptcls.dtype = self.gcl_dtype
        
        graph = self.__tographicle__(ptcls, edges)
        return graph

