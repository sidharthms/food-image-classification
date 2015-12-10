"""
Food Data dataset.
"""
import time

__authors__ = "Ian Goodfellow, Sidharth Mudgal"

import numpy as N
import stl10_input
import pdb
import gzip
import cPickle
np = N
from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
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

    def __init__(self, which_set, start=None, stop=None, center=False, rescale=False, path='./',
                 axes=('b', 0, 1, 'c')):
        # we define here:
        dtype = 'uint8'

        # we also expose the following details:
        self.img_shape = (96, 96, 3)
        self.img_size = np.prod(self.img_shape)
        self.n_classes = 10

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

        # uectrain = load_zipped_pickle(path + 'UECTestInstance.gz')
        # uectrain = np.array(uectrain).astype(dtype=np.uint8)
        # uectrain.tofile(path + 'UECTestInstance.bin')

        # print 'original length', len(uectrain)
        # pdb.set_trace()
        # save_zipped_pickle(path + 'UECTrainInstance1.gz', uectrain[:6000])
        # save_zipped_pickle(path + 'UECTrainInstance2.gz', uectrain[6000:12000])

        # uectrain = load_zipped_pickle(path + 'UECTrainInstance.gz')
        # print 'original length', len(uectrain)
        # pdb.set_trace()
        # save_zipped_pickle(path + 'UECTrainInstance1.gz', uectrain[:6000])
        # save_zipped_pickle(path + 'UECTrainInstance2.gz', uectrain[6000:12000])

        # uectest = load_zipped_pickle(path + 'UECTestInstance.gz')
        # print 'original length', len(uectest)
        # pdb.set_trace()
        # save_zipped_pickle(path + 'UECTestInstance1.gz', uectest[:1000])
        # save_zipped_pickle(path + 'UECTestInstance2.gz', uectest[1000:2000])

        # mit = load_zipped_pickle(path + 'MITTrainInstance.gz')
        # mit = np.array(mit).astype(dtype=np.uint8)
        # mit.tofile(path + 'MITTrainInstance.bin')

        # mit = load_zipped_pickle(path + 'MITTestInstance.gz')
        # mit = np.array(mit).astype(dtype=np.uint8)
        # mit.tofile(path + 'MITTestInstance.bin')

        # mittest = load_zipped_pickle(path + 'MITTrainInstance.gz')
        # print 'original length', len(mittest)
        # pdb.set_trace()
        # save_zipped_pickle(path + 'MITTrainInstance.gz', mittest[:1166])

        if which_set == 'train':
            im_path = path + 'train_images.bin'
            y_path = path + 'train_labels.bin'
        else:
            assert which_set == 'test'
            im_path = path + 'test_images.bin'
            y_path = path + 'test_labels.bin'

        time1 = time.time()
        X = np.fromfile(im_path, dtype=np.uint8).reshape((-1, 96, 96, 3))
        Y = np.fromfile(y_path, dtype=np.uint8).reshape((-1, 1))

        time2 = time.time()
        print 'Loading data took %0.3f ms' % ((time2-time1)*1000.0)

        y_labels = 2

        m, r, c, l = X.shape
        assert r == 96
        assert c == 96
        assert l == 3

        super(FoodData, self).__init__(topo_view=dimshuffle(X), y=Y, axes=axes, y_labels=y_labels)

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

        if center:
            self.X -= 127.5
        self.center = center

        if rescale:
            self.X /= 127.5
        self.rescale = rescale

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval, -1., 1.)

        return rval

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
        return FoodData(which_set='test', center=self.center,
                        rescale=self.rescale, shuffle=self.shuffle,
                        axes=self.axes)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def save_zipped_pickle(filename, obj, protocol=2):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def save_pickle(filename, obj, protocol=2):
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)
