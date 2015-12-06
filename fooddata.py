import gzip
import random
import cPickle

__author__ = 'sidharth'

"""
The MNIST dataset.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as N
import stl10_input
import pdb
np = N
from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels
from pylearn2.utils.rng import make_np_rng

class FoodData(dense_design_matrix.DenseDesignMatrix):
    """
    The MNIST dataset

    Parameters
    ----------
    which_set : str
        'train' or 'test'
    center : bool
        If True, preprocess so that each pixel has zero mean.
    shuffle : WRITEME
    binarize : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    preprocessor : WRITEME
    fit_preprocessor : WRITEME
    fit_test_preprocessor : WRITEME
    """

    def __init__(self, which_set, start=None, stop=None, path='./', axes=['b', 0, 1, 'c']):
        self.args = locals()

        if which_set not in ['train', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        def dimshuffle(b01c):
            """
            .. todo::

                WRITEME
            """
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if which_set == 'train':
            im_path1 = path + 'UECTrainInstance.gz'
            im_path2 = path + 'stl10_binary/train_X.bin'
            im_path3 = path + 'MITTrainInstance.gz'
        else:
            assert which_set == 'test'
            im_path1 = path + 'UECTestInstance.gz'
            im_path2 = path + 'stl10_binary/test_X.bin'
            im_path3 = path + 'MITTestInstance.gz'

        # Path substitution done here in order to make the lower-level
        # mnist_ubyte.py as stand-alone as possible (for reuse in, e.g.,
        # the Deep Learning Tutorials, or in another package).
        positive_data = load_zipped_pickle(im_path1)[0]
        negative_data2 = load_zipped_pickle(im_path3)
        negative_images1 = stl10_input.read_all_images(im_path2)
        negative_data1 = [negative_images1, np.zeros([len(negative_images1)])]
        pdb.set_trace()

        all_images = positive_data[0] + negative_data1[0] + negative_data2[0]
        all_labels = positive_data[1] + negative_data1[1] + negative_data2[1]
        pdb.set_trace()

        shuffled = random.shuffle(zip(all_images, all_labels))

        topo_view = [instance[0] for instance in shuffled]
        y = [instance[1] for instance in shuffled]

        y_labels = 2

        m, r, c = topo_view.shape
        assert r == 96
        assert c == 96
        pdb.set_trace()
        topo_view = topo_view.reshape(m, r, c, 1)

        # if which_set == 'train':
        #     assert m == 60000
        # elif which_set == 'test':
        #     assert m == 10000
        # else:
        #     assert False

        if shuffle:
            self.shuffle_rng = make_np_rng(
                None, [1, 2, 3], which_method="shuffle")
            for i in xrange(topo_view.shape[0]):
                j = self.shuffle_rng.randint(m)
                # Copy ensures that memory is not aliased.
                tmp = topo_view[i, :, :, :].copy()
                topo_view[i, :, :, :] = topo_view[j, :, :, :]
                topo_view[j, :, :, :] = tmp

                tmp = y[i:i + 1].copy()
                y[i] = y[j]
                y[j] = tmp

        pdb.set_trace()
        super(FoodData, self).__init__(topo_view=dimshuffle(topo_view), y=y,
                                    axes=axes, y_labels=y_labels)

        assert not N.any(N.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start

        if which_set == 'test':
            assert fit_test_preprocessor is None or \
                (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        return N.clip(X * 2. - 1., -1., 1.)

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        """
        .. todo::

            WRITEME
        """
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return MNIST(**args)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object