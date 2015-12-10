import math
import os
import numpy
import random
import pdb
import argparse
from pylearn2.config import yaml_parse
from pylearn2.monitor import read_channel
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
import sys

__author__ = 'sidharth'

sgd_seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
               str(random.randrange(1000)) + ']'
mlp_seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
               str(random.randrange(1000)) + ']'
k = 100

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
    'init_momentum': numpy.array([0.7]),
    'dropout_h2': numpy.array([9]),
    'dropout_h3': numpy.array([9]),
    'dropout_h4': numpy.array([8]),
    'dropout_y': numpy.array([8]),
}

default_args = {
    'max_norm_h3': numpy.array([1.0]),
    'max_norm_h4': numpy.array([1.0]),
    'max_norm_y': numpy.array([0.3]),
    'l_ir_h2': numpy.array([-2.7]),
    'l_ir_h3': numpy.array([-1.22]),
    'l_ir_h4': numpy.array([0.5]),
    'l_ir_y': numpy.array([-2.27]),
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


def main(job_id, params, cache):
    # Fix sub directory problems
    sys.path.append(os.path.dirname(os.getcwd()))
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Add parameters that are not currently being tuned but could potentially be tuned.
    params.update(additional_args)

    dropout_h2 = float(params['dropout_h2'][0]) / 10
    dropout_h3 = float(params['dropout_h3'][0]) / 10
    dropout_h4 = float(params['dropout_h4'][0]) / 10
    dropout_y = float(params['dropout_y'][0]) / 10
    fixed_params = (params['kernel_size_h2'][0], params['kernel_size_h3'][0], params['dropout_h2'][0],
                    params['dropout_h3'][0], params['dropout_h4'][0], params['dropout_y'][0])
    if 'cached_trainer' + str(fixed_params) not in cache:
        train_params = {
            'train_stop': 20000,
            'valid_stop': 24000,
            'test_stop': 4000,
            'batch_size': 100,
            'max_epochs': 1,
            'max_batches': 50,
            'sgd_seed': sgd_seed_str,
            'mlp_seed': mlp_seed_str,
            'save_file': 'result',

            'kernel_size_h2': int(params['kernel_size_h2'][0]),
            'output_channels_h2': 1 * k,
            'irange_h2': math.pow(10, params['l_ir_h2'][0]),
            'max_kernel_norm_h2': params['max_norm_h2'][0],
            'dropout_h2': dropout_h2,
            'dscale_h2': 1.0 / dropout_h2,
            'w_lr_sc_h2': math.pow(dropout_h2, 2),
            'weight_decay_h2': math.pow(10, params['l_wdecay_h2'][0]),

            'kernel_size_h3': int(params['kernel_size_h3'][0]),
            'output_channels_h3': int(1.7 * k),
            'irange_h3': math.pow(10, params['l_ir_h3'][0]),
            'max_kernel_norm_h3': params['max_norm_h3'][0],
            'dropout_h3': dropout_h3,
            'dscale_h3': 1.0 / dropout_h3,
            'w_lr_sc_h3': math.pow(dropout_h3, 2),
            'weight_decay_h3': math.pow(10, params['l_wdecay_h3'][0]),

            'kernel_size_h4': int(params['kernel_size_h4'][0]),
            'output_channels_h4': int(2.5 * k),
            'irange_h4': math.pow(10, params['l_ir_h4'][0]),
            'max_kernel_norm_h4': params['max_norm_h4'][0],
            'dropout_h4': dropout_h4,
            'dscale_h4': 1.0 / dropout_h4,
            'w_lr_sc_h4': math.pow(dropout_h4, 2),
            'weight_decay_h4': math.pow(10, params['l_wdecay_h4'][0]),

            'weight_decay_y': math.pow(10, params['l_wdecay_y'][0]),
            'max_col_norm_y': params['max_norm_y'][0],
            'irange_y': math.pow(10, params['l_ir_y'][0]),
            'dropout_y': dropout_y,
            'dscale_y': 1.0 / dropout_y,
            'w_lr_sc_y': math.pow(dropout_y, 2),
            'init_learning_rate': math.pow(10, params['log_init_learning_rate'][0]),
            'init_momentum': params['init_momentum'][0],
            'rectifier_left_slope': 0.2
        }

        with open('conv_fooddata_spearmint.yaml', 'r') as f:
            trainer = f.read()

        yaml_string = trainer % train_params
        train_obj = yaml_parse.load(yaml_string)

        if 'converge' in params:
            del train_obj.algorithm.termination_criterion._criteria[:]
            train_obj.extensions.append(MonitorBasedSaveBest('valid_y_misclass', 'best_model.pkl'))

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

    if 'converge' not in params:
        train_obj.algorithm.termination_criterion._criteria[0].initialize(train_obj.model)
    train_obj.main_loop(do_setup=False)
    original_misclass = read_channel(train_obj.model, misclass_channel)
    return float(original_misclass) * 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--converge', action='store_true', help='keep running until convergence')
    parser.add_argument('--gpu', type=int, help='request to use specific gpu')
    args = parser.parse_args()

    if args.gpu >= 0:
        import theano.sandbox.cuda
        theano.sandbox.cuda.use('gpu' + str(args.gpu))

    default_args.update(vars(args))
    main(0, default_args, {})
