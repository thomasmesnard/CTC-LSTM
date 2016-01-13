import logging
import random
import numpy

import cPickle

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, SequentialExampleScheme, ShuffledExampleScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]

def setup_datastream(path, batch_size, sort_batch_count, valid=False):
    A = numpy.load(os.path.join(path, ('valid_x_raw.npy' if valid else 'train_x_raw.npy')))
    B = numpy.load(os.path.join(path, ('valid_phn.npy' if valid else 'train_phn.npy')))
    C = numpy.load(os.path.join(path, ('valid_seq_to_phn.npy' if valid else 'train_seq_to_phn.npy')))

    D = [B[x[0]:x[1], 2] for x in C]

    ds = IndexableDataset({'input': A, 'output': D})
    stream = DataStream(ds, iteration_scheme=ShuffledExampleScheme(len(A)))

    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size * sort_batch_count))
    comparison = _balanced_batch_helper(stream.sources.index('input'))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)

    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size, num_examples=len(A)))
    stream = Padding(stream, mask_sources=['input', 'output'])

    return ds, stream

if __name__ == "__main__":
    ds, stream = setup_datastream(batch_size=2,
                                  path='/home/lx.nobackup/datasets/timit/readable')

    for i, d in enumerate(stream.get_epoch_iterator()):
        print '--'
        print d


        if i > 2: break

# vim: set sts=4 ts=4 sw=4 tw=0 et :
