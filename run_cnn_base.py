import math
import numpy
import random
import pdb
from pylearn2.config import yaml_parse
from pylearn2.monitor import read_channel

__author__ = 'sidharth'

seed_str = '[' + str(random.randrange(1000)) + ',' + str(random.randrange(1000)) + ',' + \
           str(random.randrange(1000)) + ']'
default_seed = [2012, 11, 6, 9]
k = 50

misclass_channel = 'valid_y_misclass'

def main(job_id, params, cache):
    if 'cached_trainer' not in cache:
        train_params = {
            'train_stop': 40000,
            'valid_stop': 50000,
            'test_stop': 10000,
            'batch_size': 100,
            'max_epochs': 2,
            'max_batches': 10,
            'sgd_seed': seed_str,
            'save_file': 'result',

            'kernel_size_h2': 5,
            'output_channels_h2': 1 * k,
            'irange_h2': math.pow(10, params['log_irange_h2'][0]),
            'max_kernel_norm_h2': params['max_kernel_norm_h2'][0],

            'weight_decay_h2': math.pow(10, params['log_weight_decay_h2'][0]),
            'weight_decay_y': math.pow(10, params['log_weight_decay_y'][0]),
            'max_col_norm_y': params['max_col_norm_y'][0],
            'irange_y': math.pow(10, params['log_irange_y'][0]),
            'init_learning_rate': math.pow(10, params['log_init_learning_rate'][0]),
            'init_momentum': params['init_momentum'][0],
            'rectifier_left_slope': 0.1
        }

        with open('conv_fooddata_spearmint.yaml', 'r') as f:
            trainer = f.read()

        yaml_string = trainer % train_params
        train_obj = yaml_parse.load(yaml_string)
        train_obj.setup()
        train_obj.model.monitor.on_channel_conflict = 'ignore'
        cache['cached_trainer'] = train_obj

    else:
        train_obj = cache['cached_trainer']
        train_obj.model.monitor.set_state([0, 0, 0])
        train_obj.model.training_succeeded = False
        # train_obj.algorithm.update_callbacks[0].reinit_from_monitor()

        model = train_obj.model
        model_params = dict([(param.name, param) for param in model.get_params()])

        irange_h2 = math.pow(10, params['log_irange_h2'][0])
        rng = numpy.random.RandomState(default_seed)
        h2_W = rng.uniform(-irange_h2, irange_h2, (model.layers[0].detector_space.num_channels,
                                                   model.layers[0].input_space.num_channels,
                                                   model.layers[0].kernel_shape[0],
                                                   model.layers[0].kernel_shape[1])).astype(numpy.float32)
        model_params['h2_W'].set_value(h2_W)
        model.layers[0].extensions[0].max_limit.set_value(params['max_kernel_norm_h2'][0].astype(numpy.float32))

        irange_y = math.pow(10, params['log_irange_y'][0])
        softmax_W = rng.uniform(-irange_y, irange_y,
                                (model.layers[1].input_dim, model.layers[1].n_classes)).astype(numpy.float32)
        model_params['softmax_W'].set_value(softmax_W)

        new_coeffs = [math.pow(10, params['log_weight_decay_h2'][0]), math.pow(10, params['log_weight_decay_y'][0])]
        for coeff, new_coeff in zip(train_obj.algorithm.cost.costs[1].coeffs, new_coeffs):
            coeff.set_value(new_coeff)
        model.layers[1].extensions[0].max_limit.set_value(params['max_col_norm_y'][0].astype(numpy.float32))

        train_obj.algorithm.learning_rate.set_value(
                math.pow(10, params['log_init_learning_rate'][0].astype(numpy.float32)))
        train_obj.algorithm.learning_rule.momentum.set_value(params['init_momentum'][0].astype(numpy.float32))
        pass

    train_obj.algorithm.termination_criterion._criteria[0].initialize(train_obj.model)
    train_obj.main_loop(do_setup=False)
    original_misclass = read_channel(train_obj.model, misclass_channel)
    return float(original_misclass)

if __name__ == "__main__":
    main(0, {
        'log_irange_h2': [-2],
        'log_irange_y': [-4],
        'max_kernel_norm_h2': [2],
        'max_col_norm_y': [2],
        'log_weight_decay_h2': [-6],
        'log_weight_decay_y': [-6],
        'log_init_learning_rate': [-4],
        'init_momentum': [0.6]
    }, {})