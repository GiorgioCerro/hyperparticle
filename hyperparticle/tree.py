import numpy as np
from graphicle import Graphicle
import graphicle as gcl
from numpy import array
from pyjet import cluster
from pyjet import PseudoJet
from numpy.lib.recfunctions import append_fields
from numpy._typing import NDArray
from typing import Tuple, Optional, Any
import operator as op


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
        self.event = self.__getevent()

    
    def __getevent(self) -> array:
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
    

    def __get1tree(self, jet: PseudoJet, first_id: int) -> Tuple[list,list]:
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
        edges = []
        #ptcl.append((jet.pt, jet.eta, jet.phi, jet.mass, first_id))

        # IMPORTANT: This edge helps to create a fully connected graph
        #edges = [(first_id, -1)]
        '''
        if not jet.parents:
            ptcl.append((jet.px, jet.py, jet.pz, jet.e, jet.id))
            edges = [(-1, jet.id)]
            return ptcl, edges
        ptcl.append((jet.px, jet.py, jet.pz, jet.e, first_id))
        edges = [(-1, first_id)]
        '''
        ptcl.append((jet.px, jet.py, jet.pz, jet.e, first_id))

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
            

    def __tographicle(self, particles: array, edges: array) -> Graphicle:
        '''Create a Graphicle object from a list of particles and a list of 
        edges between them.

        Parameters
        ----------
        particles: array
            particles of the event
        edges: array
            list of edges 

        Returns
        -------
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


    def __identify(self, jets: list, desired_jets: str = "qcd") -> NDArray:
        """Identify the jets looking at the MC truth
        """
        if desired_jets == "qcd":
            hard_graph = self.graph[self.graph.hard_mask["outgoing"]]
            jet_truth = hard_graph[np.flatnonzero(hard_graph.color.data)]
        elif desired_jets == "W":
            hard_graph = self.graph[self.graph.hard_mask]
            jet_truth = hard_graph[np.abs(hard_graph.pdg) == 24]

        reconstructed_jets = gcl.MomentumArray(
            tuple(map(op.attrgetter("px", "py", "pz", "e"), jets)))

        deltas = jet_truth.pmu.delta_R(reconstructed_jets)
        identify = np.argmin(deltas, axis=1)
        return np.unique(identify)


    def __getweights(self, graph: Graphicle) -> Graphicle: 
        '''Return an energy weighted graph
        '''
        n_dict = {graph.nodes[i]: i for i in range(len(graph.nodes))}
        weights = []
        energy = graph.pmu.data['e']
        for i in range(0, len(graph.edges), 2):
            #Graphicle object has edges in a specific order so I do this range
            a,b,c = graph.edges[i][0], graph.edges[i][1], graph.edges[i+1][1]
            e1,e2,e3 = energy[n_dict[a]], energy[n_dict[b]], energy[n_dict[c]]
            weights.append(e2/e1)
            weights.append(e3/e1)

        graph.adj.weights = weights
        return graph

        
    def history(
            self, 
            R: float, 
            p: int, 
            pt_cut: Optional[float] = None,
            eta_cut: Optional[float] = None,
            recluster: Optional[bool] = None, 
            desired_jets: str = "qcd",
            ) -> Any:
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
        if pt_cut is not None:
            jets = filter(lambda j: j.pt > pt_cut, jets)
        if eta_cut is not None:
            jets = filter(lambda j: abs(j.eta) < eta_cut, jets)
        jets = filter(lambda j: len(j.constituents()) > 4, jets)

        
        jets = list(jets) #this is not good, need to check
        if len(jets) < 1:
            return 
        identify = self.__identify(jets, desired_jets)
        jets = [jets[idx] for idx in identify]
        

        #if len(jets) < 2:
        #    if recluster is not None:
        #        constituents = []
        #        for c in jets[0].constituents():
        #            constituents.append((c.e, c.px, c.py, c.pz, c.id))
        #        constituents = np.array(constituents, dtype=self.event.dtype)
        #        sequence = cluster(constituents, R=R, p=0, ep=True)
        #        jets = sequence.inclusive_jets()
        #   
        #    first_id = -1
        #    ptcls, edges = self.__get1tree(jets[0], first_id)

        #else:
        #    ptcl = [[(jets[0].px + jets[1].py,
        #            jets[0].py + jets[1].py,
        #            jets[0].pz + jets[1].pz,
        #            jets[0].e + jets[1].e,
        #            -1)]
        #    ]
        #    edge = []
        #    first_id = -2
        #    for jet in jets:
        #        if recluster is not None:
        #            constituents = []
        #            for c in jet.constituents():
        #                constituents.append((c.e, c.px, c.py, c.pz, c.id))
        #            constituents = np.array(constituents, dtype=self.event.dtype)
        #            sequence = cluster(constituents, R=R, p=0, ep=True)
        #            jet = sequence.inclusive_jets()[0]
        #   
        #        edge.append([(-1, first_id)])
        #        ptcl_temp, edge_temp = self.__get1tree(jet, first_id)
        #        first_id = np.min(edge_temp) - 1
        #        
        #        ptcl.append(ptcl_temp)
        #        edge.append(edge_temp)

        #    edges = [item for sublist in edge for item in sublist]
        #    ptcls = [item for sublist in ptcl for item in sublist]
       

        #graph = self.__tographicle(np.array(ptcls), np.array(edges))
        graphs = []
        for jet in jets:
            if recluster is not None:
                constituents = []
                for c in jet.constituents():
                    constituents.append((c.e, c.px, c.py, c.pz, c.id))
                constituents = np.array(constituents, dtype=self.event.dtype)
                sequence = cluster(constituents, R=R, p=0, ep=True)
                jet = sequence.inclusive_jets()[0]

            first_id = -1
            ptcl, edge = self.__get1tree(jet, first_id)
            graphs.append(
                self.__tographicle(np.array(ptcl), np.array(edge)))

        return graphs

