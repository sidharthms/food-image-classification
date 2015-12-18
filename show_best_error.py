import pdb
import argparse
import numpy
import re

__author__ = 'Sidharth Mudgal'

parser = argparse.ArgumentParser()
parser.add_argument('file', help='log file')
args = parser.parse_args()

p = re.compile('\t(.*?)_misclass: (.*)')

data = numpy.ones((3, 200))

with open(args.file, 'r') as f:
    log_contents = f.read()
allmatches = p.findall(log_contents)
validation_index = [m[0] for m in allmatches[:3]].index('valid_y')

for d in xrange(3):
    accuracies = [match[1] for match in allmatches if match[0] == allmatches[d][0]]
    accuracies = numpy.array(accuracies, dtype=float)
    data[d, :len(accuracies)] = accuracies

i = data[validation_index].argmin()

for d in xrange(3):
    print allmatches[d][0], data[d, i]
