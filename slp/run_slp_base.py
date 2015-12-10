import math
import os
import numpy
import random
import pdb
from pylearn2.config import yaml_parse
from pylearn2.monitor import read_channel
import sys

__author__ = 'luke'

seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
           str(random.randrange(1000)) + ']'
default_seed = [2012, 11, 6, 9]
k = 70

additional_args = {
    'l_wdecay_y': numpy.array([-3]),
    'left_artificial_slope': ['False'],
    'right_artificial_slope': ['True'],
}

default_args = {
    'max_norm_y': numpy.array([2]),
    'l_ir_y': numpy.array([-2.16]),
    'log_init_learning_rate': numpy.array([-4])
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


def main(job_id, params, cache):
    # Fix sub directory problems
    sys.path.append(os.path.dirname(os.getcwd()))
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Add parameters that are not currently being tuned but could potentially be tuned.
    params.update(additional_args)

    train_params = {
        'train_stop': 20000,
        'valid_stop': 24000,
        'test_stop': 4000,
        'batch_size': 100,
        'max_epochs': 5,
        'max_batches': 10,
        'sgd_seed': seed_str,
        'save_file': 'result',

        'weight_decay_y': math.pow(10, params['l_wdecay_y'][0]),
        'max_col_norm_y': params['max_norm_y'][0],
        'irange_y': math.pow(10, params['l_ir_y'][0]),
        'init_learning_rate': math.pow(10, params['log_init_learning_rate'][0]),
    }

    with open('slp_fooddata.yaml', 'r') as f:
        trainer = f.read()

    print trainer

    yaml_string = trainer % train_params
    train_obj = yaml_parse.load(yaml_string)
    train_obj.setup()
    train_obj.model.monitor.on_channel_conflict = 'ignore'

    # train_obj.algorithm.termination_criterion._criteria[0].initialize(train_obj.model)
    train_obj.main_loop(do_setup=False)
    original_misclass = read_channel(train_obj.model, misclass_channel)
    return float(original_misclass)

if __name__ == "__main__":
    main(0, default_args, {})
