import os
import pdb
import argparse
import re

__author__ = 'Sidharth Mudgal'

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true', help='all datasets')
parser.add_argument('--save', help='output file', default='accuracies.csv')
parser.add_argument('files', nargs=argparse.REMAINDER, help='log files')
args = parser.parse_args()

p = re.compile('\t(.*?)_misclass: (.*)')

titles = ''
lines = ['' for x in range(200)]

for filename in args.files:
    with open(filename, 'r') as f:
        log_contents = f.read()
    allmatches = p.findall(log_contents)

    for d in xrange(3):
        if not args.all and 'test' not in allmatches[d][0]:
            continue
        titles += os.path.basename(filename) + ' - ' + allmatches[d][0] + ','

        accuracies = [match[1] for match in allmatches if match[0] == allmatches[d][0]]
        for i in xrange(200):
            if i < len(accuracies):
                lines[i] += str(accuracies[i])
            lines[i] += ','

titles += '\n'
lines = [line + '\n' for line in lines]

with open(args.save, 'w') as f:
    f.write(titles)
    f.writelines(lines)
