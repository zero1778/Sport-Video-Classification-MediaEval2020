import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='Path to input file')
    args = parser.parse_args()
    return args

def process(videoname, content):
    content.sort(key = lambda x: (x[0]))
    with open('outputs/' + videoname + '.xml', 'w') as fout:
        print('<video>', file=fout)
        for line in content:
            print('  <action begin="{}" end="{}" move="{}" />'.format(line[0], line[1], ' '.join(line[2].strip().split('_'))), file=fout)
        print('</video>', file=fout)

if not os.path.exists('outputs'):
    os.makedirs('outputs')
opt = parse_args()

data = {}

with open(opt.i, 'r') as fin:
    for line in fin:
        path, result=line.strip().split(',')
        videoname, start, end = path.split('.')[0].split('_')
        if videoname in data.keys():
            data[videoname].append([int(start), int(end), result])
        else:
            data[videoname] = [[int(start), int(end), result]]

for key, value in data.items():
    process(key, value)
