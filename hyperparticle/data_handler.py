import numpy as np
from contextlib import ExitStack
import glob
import graphicle as gcl

from torch.utils.data import Dataset

from heparchy.read.hdf import HdfReader

class ParticleDataset(Dataset):
    '''Particles shower dataset.'''

    def __init__(self, path):
        self.__files = glob.glob(path + '/*.hdf5')
        self.__ranges = [(-1, -1)]

        stack = ExitStack()
        for file in self.__files:
            file_obj = stack.enter_context(HdfReader(path=file))
            try:
                process = file_obj['signal']
                self.__process_name = 'signal'
            except KeyError:
                process = file_obj['background']
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
            process = hep_file[self.__process_name]
            _event = process[_event_idx]

            graph = gcl.Graphicle.from_numpy(
                edges = _event.edges,
                pmu = _event.pmu,
                pdg = _event.pdg,
                status = _event.status,
                final = _event.masks['final']
            )

            event_dict['MC_graph'] = graph
            for k in range(3):
                pmu = _event.custom[self.algo[k] + '_pmu']
                edges = _event.custom[self.algo[k] + '_edges']
                hyp = _event.custom[self.algo[k] + '_hyp']
                mask = _event.custom[self.algo[k] + '_mask']
                
                g = gcl.Graphicle.from_numpy(
                    edges = edges[mask][:-1],
                    pmu = pmu[mask],
                    pdg = np.ones(sum(mask)),
                )

                g.final.data = g.nodes >= 0

                event_dict[self.algo[k] + '_graph'] = g
                event_dict[self.algo[k] + '_hyp'] = hyp[mask]
                
        return event_dict



class OneSet:
    def __init__(self, file):
        self.file = file 
        self.algo = ['aKt', 'CA', 'Kt']
        with HdfReader(path=self.file) as hep_file:
            try: process = hep_file['signal']
            except KeyError: process = hep_file['background']
            self.number_of_events = len(process)
         

    def __len__(self):
        return self.number_of_events


    def __getitem__(self, idx):
        event_dict = {}
        with HdfReader(path=self.file) as hep_file:
            try: process = hep_file['signal']
            except KeyError: process = hep_file['background']
            _event = process[idx]

            graph = gcl.Graphicle.from_numpy(
                edges = _event.edges,
                pmu = _event.pmu,
                pdg = _event.pdg,
                status = _event.status,
                final = _event.masks['final']
            )

            event_dict['MC_graph'] = graph
            for k in range(1):
                pmu = _event.custom[self.algo[k] + '_pmu']
                edges = _event.custom[self.algo[k] + '_edges']
                hyp = _event.custom[self.algo[k] + '_hyp']
                mask = _event.custom[self.algo[k] + '_mask']

                g = gcl.Graphicle.from_numpy(
                    edges = edges[mask][:-1],
                    pmu = pmu[mask],
                    pdg = np.ones(sum(mask)),
                )

                g.final.data = g.nodes >= 0

                event_dict[self.algo[k] + '_graph'] = g
                event_dict[self.algo[k] + '_hyp'] = hyp[mask]
                
        return event_dict


