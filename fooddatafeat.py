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
from pylearn2.datasets import vector_spaces_dataset
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import VectorSpace, CompositeSpace, IndexSpace
from pylearn2.utils.rng import make_np_rng

class FoodData(vector_spaces_dataset.VectorSpacesDataset):
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
                 axes=('b', 'c')):

        # we also expose the following details:
        self.shape = (128)
        self.size = np.prod(self.shape)
        self.n_classes = 2

        if which_set not in ['train', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        def dimshuffle(bc):
            """
            .. todo::

                WRITEME
            """
            default = ('b', 'c')
            return bc.transpose(*[default.index(axis) for axis in axes])

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
            im_path = path + 'trainSIFT_vectors.npy'
            y_path = path + 'trainSIFT_labels.npy'
        else:
            assert which_set == 'test'
            im_path = path + 'testSIFT_vectors.npy'
            y_path = path + 'testSIFT_labels.npy'

        time1 = time.time()
        X = np.load(im_path).reshape((-1,128))
        Y = np.load(y_path).reshape((-1, 1))

        time2 = time.time()
        print 'Loading data took %0.3f ms' % ((time2-time1)*1000.0)

        y_labels = 2

        m, r = X.shape
        assert r == 128

        source = ('features', 'targets')
        space = CompositeSpace([
            VectorSpace(128),
            IndexSpace(dim = 1, max_labels = 2)
        ])

        super(FoodData, self).__init__(
            data=(X, Y),
            data_specs=(space, source)
        )

        self.X = self.data[0]
        self.y = self.data[1]

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

