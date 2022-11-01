import numpy as np
from contextlib import ExitStack
import glob
import graphicle as gcl

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from heparchy.read.hdf import HdfReader
from tree import FamilyTree

class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(self, path):
        self.__files = glob.glob(path + '/*.hdf5')
        self.__ranges = [(-1, -1)]

        stack = ExitStack()
        for file in self.__files:
            file_obj = stack.enter_context(HdfReader(path=file))
            try:
                process = file_obj.read_process(name = 'signal')
                self.__process_name = 'signal'
            except KeyError:
                process = file_obj.read_process(name = 'background')
                self.__process_name = 'background'
            ini = self.__ranges[-1][1] + 1
            fin = ini + len(process) - 1
            self.__ranges.append((ini, fin))
        stack.close()

        _dtype = [('ini', 'i4'), ('fin', 'i4')]
        self.__ranges = np.array(self.__ranges[1:], dtype=_dtype)
        
        self.algo = ['aKt', 'CA', 'Kt']


    def __len__(self):
        return self.__ranges['fin'][-1]


    def __getitem__(self, idx):
        _file_idx = int(np.where(
            np.logical_and(np.less_equal(self.__ranges['ini'], idx),
                            np.greater_equal(self.__ranges['fin'], idx)))[0])

        _event_idx = idx - self.__ranges[_file_idx]['ini']
        event_dict = {}
        with HdfReader(path=self.__files[_file_idx]) as hep_file:
            process = hep_file.read_process(name=self.__process_name)
            _event = process.read_event(_event_idx)

            graph = gcl.Graphicle.from_numpy(
                edges = _event.edges,
                pmu = _event.pmu,
                pdg = _event.pdg,
                status = _event.status,
                final = _event.mask('final')
            )

            #graph.adj = gcl.transform.particle_as_node(graph.adj)
            #graph_hyper = _event.get_custom('MC_hyp')
            event_dict['MC_graph'] = graph
            #event_dict['MC_hyp'] = graph_hyper

            for k in range(3):
                pmu = _event.get_custom(self.algo[k] + '_pmu')
                edges = _event.get_custom(self.algo[k] + '_edges')
                hyp = _event.get_custom(self.algo[k] + '_hyp')
                mask = _event.get_custom(self.algo[k] + '_mask')

                g = gcl.Graphicle.from_numpy(
                    edges = edges[mask][:-1],
                    pmu = pmu[mask],
                    pdg = np.ones(sum(mask)),
                )
                finals = g.nodes >= 0
                g.final.data = finals

                event_dict[self.algo[k] + '_graph'] = g
                event_dict[self.algo[k] + '_hyp'] = hyp[mask]
        return event_dict
