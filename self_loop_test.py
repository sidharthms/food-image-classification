__author__ = 'Sidharth Mudgal'

import os
import sys
import argparse
import theano.sandbox.cuda
import cPickle
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('exp_path', help='experiment path')
parser.add_argument('main', help='experiment path')
parser.add_argument('gpu', help='gpu to use')
args = parser.parse_args()

theano.sandbox.cuda.use('gpu' + str(args.gpu))
os.chdir(args.exp_path)
sys.path.append(os.getcwd())

module = __import__(args.main)

cache = {}

with open('results.pkl', 'wb') as f:
    cPickle.dump([], f)

while True:
    print 'Running main...'
    result = module.main(0, module.default_args, cache)
    print 'Obtained result:', result

    with open('results.pkl', 'rb') as f:
        results_list = cPickle.load(f)

    results_list.append(result)
    with open('results.pkl', 'wb') as f:
        cPickle.dump(results_list, f)
