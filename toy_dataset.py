import logging
import random
import numpy

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, SequentialExampleScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class ToyDataset(Dataset):
    def __init__(self, nb_examples, rng_seed, min_out_len, max_out_len, **kwargs):
        self.provides_sources = ('input', 'output')

        random.seed(rng_seed)

        table = [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 1, 0],
                [4, 3, 2, 3, 4],
                [4, 3, 2, 1, 0]
        ]
        prob0 = 0.7
        prob = 0.2

        self.data = []
        for n in range(nb_examples):
            o = []
            i = []
            l = random.randrange(min_out_len, max_out_len)
            for p in range(l):
                o.append(random.randrange(len(table)))
                for x in table[o[-1]]:
                    q = 0
                    if random.uniform(0, 1) < prob0:
                        i.append(x)
                    while random.uniform(0, 1) < prob:
                        i.append(x)
            self.data.append((i, o))

        super(ToyDataset, self).__init__(**kwargs)


    def get_data(self, state=None, request=None):
        if request is None:
            raise ValueError("Request required")

        return self.data[request]

# -------------- DATASTREAM SETUP --------------------

def setup_datastream(batch_size, **kwargs):
    ds = ToyDataset(**kwargs)
    stream = DataStream(ds, iteration_scheme=SequentialExampleScheme(kwargs['nb_examples']))

    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))
    stream = Padding(stream, mask_sources=['input', 'output'])

    return ds, stream

if __name__ == "__main__":

    ds, stream = setup_datastream(nb_examples=5,
                                  rng_seed=123,
                                  min_out_len=3,
                                  max_out_len=6)

    for i, d in enumerate(stream.get_epoch_iterator()):
        print '--'
        print d


        if i > 2: break

# vim: set sts=4 ts=4 sw=4 tw=0 et :
