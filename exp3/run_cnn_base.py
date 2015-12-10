import math
import os
import numpy
import random
import pdb
import argparse
from pylearn2.config import yaml_parse
from pylearn2.monitor import read_channel
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.utils import serial
import sys

__author__ = 'sidharth'

sgd_seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
               str(random.randrange(1000)) + ']'
mlp_seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
               str(random.randrange(1000)) + ']'

additional_args = {
    'max_norm_h2': numpy.array([1.0]),
    'l_wdecay_h2': numpy.array([-6]),
    'l_wdecay_h3': numpy.array([-6]),
    'l_wdecay_h4': numpy.array([-6]),
    'l_wdecay_y': numpy.array([-6]),
    'kernel_size_h2': ['5'],
    'kernel_size_h3': ['5'],
    'kernel_size_h4': ['5'],
    'log_init_learning_rate': numpy.array([-4.5]),
    'init_momentum': numpy.array([0.7])
}

# default_args = {
#     'kernel_config': ['b'],
#     'max_norm_h3': numpy.array([3.962]),
#     'max_norm_h4': numpy.array([1.802]),
#     'max_norm_y': numpy.array([0.3]),
#     'l_ir_h2': numpy.array([-1.171]),
#     'l_ir_h3': numpy.array([-1.086]),
#     'l_ir_h4': numpy.array([-1.716]),
#     'l_ir_y': numpy.array([-5.8])
# }

default_args = {
    'start': 0,
    'stop': 20000,
    'kernel_config': ['c'],
    'max_norm_h3': numpy.array([3.247]),
    'max_norm_h4': numpy.array([1.223]),
    'max_norm_y': numpy.array([0.3]),
    'l_ir_h2': numpy.array([-1.142]),
    'l_ir_h3': numpy.array([-1.012]),
    'l_ir_h4': numpy.array([-1.713]),
    'l_ir_y': numpy.array([-5.043])
}
misclass_channel = 'valid_y_misclass'


def update_conv_layer(layer, log_range, norm, model_params, rng):
    irange = math.pow(10, log_range)
    W = rng.uniform(-irange, irange, (layer.detector_space.num_channels,
                                      layer.input_space.num_channels,
                                      layer.kernel_shape[0],
                                      layer.kernel_shape[1])).astype(numpy.float32)
    model_params[layer.layer_name + '_W'].set_value(W)
    if layer.tied_b:
        model_params[layer.layer_name + '_b'].set_value(numpy.zeros(layer.detector_space.num_channels)
                                                        .astype(numpy.float32) + layer.init_bias)
    else:
        model_params[layer.layer_name + '_b'].set_value(layer.detector_space.get_origin() + layer.init_bias)

    if norm is not None:
        layer.extensions[0].max_limit.set_value(norm.astype(numpy.float32))


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

    if params['kernel_config'][0] == 'a':
        output_channels_h2 = int(2.89 * 100)
        output_channels_h3 = int(1.70 * 100)
        output_channels_h4 = int(1.00 * 100)
    elif params['kernel_config'][0] == 'b':
        output_channels_h2 = int(1.00 * 100)
        output_channels_h3 = int(1.70 * 100)
        output_channels_h4 = int(2.89 * 100)
    elif params['kernel_config'][0] == 'c':
        output_channels_h2 = int(1.00 * 50)
        output_channels_h3 = int(3.42 * 50)
        output_channels_h4 = int(11.67 * 50)
    elif params['kernel_config'][0] == 'd':
        output_channels_h2 = int(11.67 * 50)
        output_channels_h3 = int(3.42 * 50)
        output_channels_h4 = int(1.00 * 50)
    else:
        raise RuntimeError('Unknown kernel config')

    if params['rate'] is not None:
        params['log_init_learning_rate'][0] = numpy.array([params['rate']])

    fixed_params = (params['kernel_size_h2'][0], params['kernel_size_h3'][0], params['kernel_config'][0])
    if 'cached_trainer' + str(fixed_params) not in cache:
        train_params = {
            'train_start': params['start'],
            'train_stop': params['stop'],
            'valid_start': 20000,
            'valid_stop': 24000,
            'test_stop': 4000,
            'batch_size': 100,
            'max_epochs': 1,
            'max_batches': 10,
            'sgd_seed': sgd_seed_str,
            'mlp_seed': mlp_seed_str,

            'kernel_size_h2': int(params['kernel_size_h2'][0]),
            'output_channels_h2': output_channels_h2,
            'irange_h2': math.pow(10, params['l_ir_h2'][0]),
            'max_kernel_norm_h2': params['max_norm_h2'][0],

            'kernel_size_h3': int(params['kernel_size_h3'][0]),
            'output_channels_h3': output_channels_h3,
            'irange_h3': math.pow(10, params['l_ir_h3'][0]),
            'max_kernel_norm_h3': params['max_norm_h3'][0],

            'kernel_size_h4': int(params['kernel_size_h4'][0]),
            'output_channels_h4': output_channels_h4,
            'irange_h4': math.pow(10, params['l_ir_h4'][0]),
            'max_kernel_norm_h4': params['max_norm_h4'][0],

            'weight_decay_h2': math.pow(10, params['l_wdecay_h2'][0]),
            'weight_decay_h3': math.pow(10, params['l_wdecay_h3'][0]),
            'weight_decay_h4': math.pow(10, params['l_wdecay_h4'][0]),
            'weight_decay_y': math.pow(10, params['l_wdecay_y'][0]),
            'max_col_norm_y': params['max_norm_y'][0],
            'irange_y': math.pow(10, params['l_ir_y'][0]),
            'init_learning_rate': math.pow(10, params['log_init_learning_rate'][0]),
            'init_momentum': params['init_momentum'][0],
            'rectifier_left_slope': 0.2
        }

        with open('conv_fooddata_spearmint.yaml', 'r') as f:
            trainer = f.read()

        yaml_string = trainer % train_params
        train_obj = yaml_parse.load(yaml_string)

        if 'converge' in params:
            train_obj.algorithm.termination_criterion._criteria[0]._max_epochs = 100
            train_obj.extensions.append(MonitorBasedSaveBest('valid_y_misclass', params['save']))

        train_obj.setup()
        train_obj.model.monitor.on_channel_conflict = 'ignore'
        # cache['cached_trainer' + str(fixed_params)] = train_obj

    else:
        train_obj = cache['cached_trainer' + str(fixed_params)]
        train_obj.model.monitor.set_state([0, 0, 0])
        train_obj.model.training_succeeded = False
        # train_obj.algorithm.update_callbacks[0].reinit_from_monitor()

        model = train_obj.model
        model_params = dict([(param.name, param) for param in model.get_params()])

        rng = model.rng

        update_conv_layer(model.layers[0], params['l_ir_h2'][0], params['max_norm_h2'][0], model_params, rng)
        update_conv_layer(model.layers[1], params['l_ir_h3'][0], params['max_norm_h3'][0], model_params, rng)
        update_conv_layer(model.layers[2], params['l_ir_h4'][0], params['max_norm_h4'][0], model_params, rng)
        update_softmax_layer(model.layers[3], params['l_ir_y'][0], params['max_norm_y'][0], model_params, rng)

        train_obj.algorithm.learning_rate.set_value(
                math.pow(10, params['log_init_learning_rate'][0].astype(numpy.float32)))
        train_obj.algorithm.learning_rule.momentum.set_value(params['init_momentum'][0].astype(numpy.float32))
        pass

    pretrained_model_path = params.get('model', None)
    if pretrained_model_path is not None:
        print 'loading pre trained model'
        pretrained_model = serial.load(pretrained_model_path)
        print 'loading done'
        train_obj.model.set_param_values(pretrained_model.get_param_values())

    if 'converge' not in params:
        train_obj.algorithm.termination_criterion._criteria[0].initialize(train_obj.model)
    train_obj.main_loop(do_setup=False)
    original_misclass = read_channel(train_obj.model, misclass_channel)
    return float(original_misclass) * 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--converge', action='store_true', help='keep running until convergence')
    parser.add_argument('--model', help='load pre trained model')
    parser.add_argument('--start', help='start for training set', type=int, default=0)
    parser.add_argument('--stop', help='stop for training set', type=int, default=20000)
    parser.add_argument('--gpu', type=int, help='request to use specific gpu')
    parser.add_argument('--save', default='best_model.pkl', help='file to save best model to')
    parser.add_argument('--rate', type=float, help='learning rate')
    args = parser.parse_args()

    if args.gpu >= 0:
        import theano.sandbox.cuda
        theano.sandbox.cuda.use('gpu' + str(args.gpu))

    default_args.update(vars(args))
    main(0, default_args, {})
