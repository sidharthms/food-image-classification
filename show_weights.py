import sys
import os
import pdb
import random
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.rng import make_np_rng
from pylearn2.space import CompositeSpace
from theano import tensor as T
from fooddata import FoodData
import numpy as np
import argparse
import theano
from theano import tensor as T

__author__ = 'Sidharth Mudgal'

theano.config.compute_test_value = 'pdb'


def compile_theano_function(model, layer_batches):
    input_space = model.get_input_space()
    X = input_space.make_theano_batch(name='show_weights_X', batch_size=model.batch_size)
    raw_activations = model.fprop(X, return_all=True)
    activations = [layer_acts.dimshuffle([1, 0, 2, 3]) for layer_acts in raw_activations[:-1]] + \
                  [raw_activations[-1].dimshuffle([1, 0])]
    activations = [activations[l].reshape((layer_batches[l].shape[0],
                                           layer_batches[l].size / layer_batches[l].shape[0]))
                   for l in xrange(len(model.layers))]
    max_idxes = [activations[l].argmax(1) for l in xrange(len(model.layers))]
    # max_idxes[-1] = max_idxes[-1].dimshuffle(0, 'x', 'x')
    maxes = [activations[l].max(1) for l in xrange(len(model.layers))]

    return theano.function([X], outputs=[T.concatenate(max_idxes), T.concatenate(maxes)])

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='load trained model', default='model.pkl')
parser.add_argument('--gpu', type=int, help='request to use specific gpu')
args = parser.parse_args()

if args.gpu >= 0:
    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu' + str(args.gpu))

model = serial.load(args.model)
dataset = FoodData(which_set='train', start=0, stop=10000)

iterator = dataset.iterator(mode='even_sequential',
                            batch_size=model.batch_size,
                            data_specs=(model.get_input_space(), model.input_source),
                            return_tuple=True)

layer_batches = [layer.get_output_space().get_origin_batch(model.batch_size) for layer in model.layers]
layer_batches[:-1] = [layer_batch.transpose([1, 0, 2, 3]) for layer_batch in layer_batches[:-1]]
layer_batches[-1] = layer_batches[-1].transpose([1, 0])
get_activations_argmax = compile_theano_function(model, layer_batches)

channels = [layer_batches[l].shape[0] for l in xrange(len(model.layers))]
patch_size = [layer_batches[l].shape[2] for l in xrange(len(model.layers) - 1)]
max_activations = [[[] for c in range(channels[l])] for l in range(len(model.layers))]

for bi, batch in enumerate(iterator):
    cranges = [0]
    for l in xrange(len(channels)):
        cranges.append(channels[l] + cranges[l])

    max_idxes, maxes = get_activations_argmax(batch[0])
    maxes = [maxes[cranges[l]:cranges[l+1]] for l in xrange(len(model.layers))]
    maxact_vect = [np.column_stack(np.unravel_index(max_idxes[cranges[l]:cranges[l+1]], layer_batches[l].shape[1:]))
                   for l in xrange(len(model.layers))]
    maxact_vect = [[(v[0] + bi * model.batch_size, v[1], v[2]) for v in maxact_vect[l]]
                   for l in xrange(len(model.layers)-1)]
    for l in xrange(len(model.layers)-1):
        for c in xrange(channels[l]):
            max_activations[l][c].append((maxact_vect[l][c], maxes[l][c]))
pdb.set_trace()


