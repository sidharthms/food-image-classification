import sys
import os
import pdb
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import numpy as np
import argparse

# Fix sub directory problems
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def predict(model_path):
    try:
        test_X_path = "../testSIFT_vectors.npy"
        test_pick_path = "../testSIFT_picids.npy"
        test_y_path = "../test_labels.npy"

    except IndexError:
        print "Usage: predict.py <model file> <test file> <output file>"
        quit()

    try:
        model = serial.load( model_path )
    except Exception, e:
        print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
        print e

    datasetX = np.load(test_X_path).reshape((-1,128)).astype(dtype=np.float32)
    dataPic = np.load(test_pick_path).reshape((-1,1))
    datasetY = np.load(test_y_path).reshape((-1,1))

    batch_size = 100
    model.set_batch_size(batch_size)


    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)


    from theano import tensor as T

    Y = T.argmax(Y, axis=1)

    from theano import function

    f = function([X], Y)
    print("loading data and predicting...")

    y = []
    for i in xrange(datasetX.shape[0] / batch_size):
        x = datasetX[i*batch_size:(i+1)*batch_size,:]
        y.extend(f(x).tolist())
        # pdb.set_trace()

    print("writing predictions...")
    assert dataPic.shape[0] == datasetX.shape[0]

    # pdb.set_trace()


    count = [0] * datasetY.shape[0]
    for i in xrange(len(y)):
        if(y[i] == 0):
            count[dataPic[i]] -= 4
        else:
            count[dataPic[i]] += 1

    # print(y)

    for i in xrange(len(count)):
        if(count[i] >= 0):
            count[i] = 1
        if(count[i] < 0):
            count[i] = 0

    correct = 0
    for i in xrange(len(count)):
        if(count[i] == datasetY[i][0]):
            correct += 1

    return (len(count) - float(correct))/len(count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='load trained model', default='model.pkl')
    args = parser.parse_args()
    print predict(args.model)
