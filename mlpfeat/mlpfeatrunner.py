import argparse
import multiprocessing
import os
import sys
import pdb

__author__ = 'sidharth'

parser = argparse.ArgumentParser()
parser.add_argument('--predict', help='load trained model', action='store_true')
parser.add_argument('--file', help='load trained model', default='mlpfeatacc.csv')
args = parser.parse_args()

def runpredict(modelpath):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # pdb.set_trace()
    import pridict_mlp
    print 'predicing for path', modelpath
    return pridict_mlp.predict(modelpath)

pool = multiprocessing.Pool(16)

if args.predict:
    paths = []
    for e in xrange(1, 5):
        for t in xrange(1, 5):
            paths.append('mlpfeat/best_model_' + str(t * 5) + '_' + str(e * 50) + '.pkl')
    results = pool.map(runpredict, paths)
    # print runpredict(paths[0])
    with open(args.file, 'w') as f:
        for e in xrange(4):
            line = ''
            for t in xrange(4):
                line += str(results[e * 4 + t]) + ','
            f.write(line + '\n')

else:
    commands = []
    def runcommand(command):
        print 'executing', command
        os.system(command)
    for t in xrange(1, 5):
        for e in xrange(1, 5):
            command = ('python -u run_mlp_base.py --converge --gpu ' + str(t % 2) +
                       ' --save best_model_' + str(t * 5) + '_' + str(e * 50) + '.pkl --stop ' + str(t * 5000) +
                       ' --epochs ' + str(e * 50) + ' >&loop' + str(t * 5) + '_' + str(e * 50) + '.log')
            commands.append(command)

    pool.map(runcommand, commands)
