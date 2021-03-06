import math
import os
import numpy
import random
import pdb
import argparse
from pylearn2.config import yaml_parse
from pylearn2.monitor import read_channel
from pylearn2.utils import serial
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
import sys

__author__ = 'luke'

seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
           str(random.randrange(1000)) + ']'

additional_args = {
    'l_wdecay_y': numpy.array([-5]),
    'l_wdecay_h2': numpy.array([-5]),
    'l_wdecay_h3': numpy.array([-5]),
    'start': 0,
    'stop': 20000,
}

default_args = {
    'dim_h1': ['9160'],
    'dim_h2': ['1620'],
    'max_norm_h1': numpy.array([2.5]),
    'max_norm_h2': numpy.array([10.0]),
    'max_norm_y': numpy.array([3.737]),
    'l_ir_h1': numpy.array([-4.778]),
    'l_ir_h2': numpy.array([0.331]),
    'l_ir_y': numpy.array([-2.092]),
    'log_init_learning_rate': numpy.array([-5.684])
}

misclass_channel = 'valid_y_misclass'


def update_softmax_layer(layer, log_range, norm, model_params, rng):
    assert layer.layer_name == 'y'
    irange = math.pow(10, log_range)
    softmax_W = rng.uniform(-irange, irange, (layer.input_dim, layer.n_classes)).astype(numpy.float32)
    model_params['softmax_W'].set_value(softmax_W)
    model_params['softmax_b'].set_value(numpy.zeros((layer.n_classes - layer.non_redundant,)).astype(numpy.float32))
    if norm is not None:
        layer.extensions[0].max_limit.set_value(norm.astype(numpy.float32))


def main(job_id, requested_params, cache):
    # Fix sub directory problems
    sys.path.append(os.path.dirname(os.getcwd()))
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Add parameters that are not currently being tuned but could potentially be tuned.
    params = additional_args
    params.update(requested_params)

    if params.get('rate', None) is not None:
        params['log_init_learning_rate'][0] += numpy.array([params['rate']])

    train_params = {
        'train_start': params['start'],
        'train_stop': params['stop'],
        'valid_start': 20000,
        'valid_stop': 24000,
        'test_stop': 4000,
        'batch_size': 100,
        'max_epochs': params.get('epochs', 5),
        'max_batches': 10,
        'sgd_seed': seed_str,

        'dim_h1': int(params['dim_h1'][0]),
        'irange_h1': math.pow(10, params['l_ir_h1'][0]),
        'max_col_norm_h1': params['max_norm_h1'][0],

        'dim_h2': int(params['dim_h2'][0]),
        'irange_h2': math.pow(10, params['l_ir_h2'][0]),
        'max_col_norm_h2': params['max_norm_h2'][0],

        'weight_decay_h2': math.pow(10, params['l_wdecay_h2'][0]),
        'weight_decay_h3': math.pow(10, params['l_wdecay_h3'][0]),
        'weight_decay_y': math.pow(10, params['l_wdecay_y'][0]),
        'max_col_norm_y': params['max_norm_y'][0],
        'irange_y': math.pow(10, params['l_ir_y'][0]),
        'init_momentum': 0.5,
        'init_learning_rate': math.pow(10, params['log_init_learning_rate'][0]),
    }

    with open('mlp_fooddata.yaml', 'r') as f:
        trainer = f.read()

    yaml_string = trainer % train_params
    train_obj = yaml_parse.load(yaml_string)

    pretrained_model_path = params.get('model', None)
    if pretrained_model_path is not None:
        print 'loading pre trained model'
        pretrained_model = serial.load(pretrained_model_path)
        print 'loading done'
        train_obj.model.set_param_values(pretrained_model.get_param_values())

    if 'converge' in params:
        train_obj.algorithm.termination_criterion._criteria[0]._max_epochs = params.get('epochs', 100)
        # train_obj.extensions.append(MonitorBasedSaveBest('valid_y_misclass', 'best_model.pkl'))

    train_obj.setup()
    train_obj.model.monitor.on_channel_conflict = 'ignore'
    if 'converge' not in params:
        train_obj.algorithm.termination_criterion._criteria[0].initialize(train_obj.model)
    train_obj.main_loop(do_setup=False)
    if 'converge' in params:
        print 'saving model'
        serial.save(params['save'] + 'f', train_obj.model, on_overwrite='backup')
    original_misclass = read_channel(train_obj.model, misclass_channel)
    return float(original_misclass)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--converge', action='store_true', help='keep running until convergence')
    parser.add_argument('--model', help='load pre trained model')
    parser.add_argument('--start', help='start for training set', type=int, default=0)
    parser.add_argument('--stop', help='stop for training set', type=int, default=20000)
    parser.add_argument('--gpu', type=int, help='request to use specific gpu')
    parser.add_argument('--save', default='best_model.pkl', help='file to save best model to')
    parser.add_argument('--rate', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='num epochs')
    args = parser.parse_args()

    if args.gpu >= 0:
        import theano.sandbox.cuda
        theano.sandbox.cuda.use('gpu' + str(args.gpu))

    filtered_args = {k: v for k, v in vars(args).items() if v is not None}
    default_args.update(filtered_args)
    main(0, default_args, {})
