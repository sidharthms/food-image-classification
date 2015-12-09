import gzip
import cPickle
import os
import argparse
import numpy as np
import stl10_input
from pylearn2.utils.rng import make_np_rng

__author__ = 'sidharth'

parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to data files')
args = parser.parse_args()

os.chdir(args.path)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def shuffle(X, Y):
    shuffle_rng = make_np_rng(
        None, [1, 2, 3], which_method="shuffle")
    for i in xrange(X.shape[0]):
        j = shuffle_rng.randint(len(X))
        # Copy ensures that memory is not aliased.
        tmp = X[i, :, :, :].copy()
        X[i, :, :, :] = X[j, :, :, :]
        X[j, :, :, :] = tmp

        tmp = Y[i:i + 1].copy()
        Y[i] = Y[j]
        Y[j] = tmp

TrainX = np.zeros((24000, 96, 96, 3), dtype=np.uint8)
TrainY = np.zeros((24000, 1), dtype=np.uint8)

s = 0
TrainX[s:s + 12000] = np.array(load_zipped_pickle('UECTrainInstance.gz')).astype(dtype=np.uint8)
TrainY[s:s + 12000] = np.ones([12000, 1])

s += 12000
TrainX[s:s + 5000] = np.array(stl10_input.read_all_images('stl10_binary/train_X.bin')[:5000]).astype(dtype=np.uint8)
TrainY[s:s + 5000] = np.zeros([5000, 1])

s += 5000
TrainX[s:s + 7000] = np.array(load_zipped_pickle('MITTrainInstance.gz')).astype(dtype=np.uint8)
TrainY[s:s + 7000] = np.zeros([7000, 1])

shuffle(TrainX, TrainY)

TrainX.tofile('train_images.bin')
TrainY.tofile('train_labels.bin')

TestX = np.zeros((4000, 96, 96, 3), dtype=np.uint8)
TestY = np.zeros((4000, 1), dtype=np.uint8)

s = 0
TestX[s:s + 2000] = np.array(load_zipped_pickle('UECTestInstance.gz')).astype(dtype=np.uint8)
TestY[s:s + 2000] = np.ones([2000, 1])

s += 2000
TestX[s:s + 834] = np.array(stl10_input.read_all_images('stl10_binary/test_X.bin')[:834]).astype(dtype=np.uint8)
TestY[s:s + 834] = np.zeros([834, 1])

s += 834
TestX[s:s + 1166] = np.array(load_zipped_pickle('MITTestInstance.gz')).astype(dtype=np.uint8)
TestY[s:s + 1166] = np.zeros([1166, 1])

shuffle(TestX, TestY)

TestX.tofile('test_images.bin')
TestY.tofile('test_labels.bin')
