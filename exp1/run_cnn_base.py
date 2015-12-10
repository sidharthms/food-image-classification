import math
import os
import numpy
import random
import pdb
from pylearn2.config import yaml_parse
from pylearn2.monitor import read_channel
import sys

__author__ = 'sidharth'

sgd_seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
               str(random.randrange(1000)) + ']'
mlp_seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
               str(random.randrange(1000)) + ']'
k = 70

additional_args = {
    'l_wdecay_h2': numpy.array([-4]),
    'l_wdecay_h3': numpy.array([-5]),
    'l_wdecay_y': numpy.array([-3]),
    'left_artificial_slope': ['False'],
    'right_artificial_slope': ['True'],
    'kernel_size_h2': ['5'],
    'kernel_size_h3': ['5'],
}

default_args = {
    'max_norm_h2': numpy.array([1.09]),
    'max_norm_h3': numpy.array([1.09]),
    'max_norm_y': numpy.array([2]),
    'l_ir_h2': numpy.array([-1.55]),
    'l_ir_h3': numpy.array([-1.55]),
    'l_ir_y': numpy.array([-2.16]),
    'log_init_learning_rate': numpy.array([-4])
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

    fixed_params = (params['kernel_size_h2'][0], params['kernel_size_h3'][0])
    if 'cached_trainer' + str(fixed_params) not in cache:
        train_params = {
            'train_stop': 20000,
            'valid_stop': 24000,
            'test_stop': 4000,
            'batch_size': 100,
            'max_epochs': 3,
            'max_batches': 10,
            'sgd_seed': sgd_seed_str,
            'mlp_seed': mlp_seed_str,
            'save_file': 'result',

            'kernel_size_h2': int(params['kernel_size_h2'][0]),
            'output_channels_h2': 1 * k,
            'irange_h2': math.pow(10, params['l_ir_h2'][0]),
            'max_kernel_norm_h2': params['max_norm_h2'][0],

            'kernel_size_h3': int(params['kernel_size_h3'][0]),
            'output_channels_h3': int(1.5 * k),
            'irange_h3': math.pow(10, params['l_ir_h3'][0]),
            'max_kernel_norm_h3': params['max_norm_h3'][0],

            'weight_decay_h2': math.pow(10, params['l_wdecay_h2'][0]),
            'weight_decay_h3': math.pow(10, params['l_wdecay_h3'][0]),
            'weight_decay_y': math.pow(10, params['l_wdecay_y'][0]),
            'max_col_norm_y': params['max_norm_y'][0],
            'irange_y': math.pow(10, params['l_ir_y'][0]),
            'init_learning_rate': math.pow(10, params['log_init_learning_rate'][0]),
            'init_momentum': 0.5,
            'rectifier_left_slope': 0.2
        }

        with open('conv_fooddata_spearmint.yaml', 'r') as f:
            trainer = f.read()

        yaml_string = trainer % train_params
        train_obj = yaml_parse.load(yaml_string)
        train_obj.setup()
        train_obj.model.monitor.on_channel_conflict = 'ignore'
        cache['cached_trainer' + str(fixed_params)] = train_obj

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
        update_softmax_layer(model.layers[2], params['l_ir_y'][0], params['max_norm_y'][0], model_params, rng)

        train_obj.algorithm.learning_rate.set_value(
                math.pow(10, params['log_init_learning_rate'][0].astype(numpy.float32)))
        # train_obj.algorithm.learning_rule.momentum.set_value(params['init_momentum'][0].astype(numpy.float32))
        pass

    train_obj.algorithm.termination_criterion._criteria[0].initialize(train_obj.model)
    train_obj.main_loop(do_setup=False)
    original_misclass = read_channel(train_obj.model, misclass_channel)
    return float(original_misclass) * 50

if __name__ == "__main__":
    main(0, default_args, {})
