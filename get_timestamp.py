#!/usr/bin/env python

"""
Given two values, looks in a 3 column-file (output of sham.pl)
which time frame matches closest.
"""

import sys
import os
from subprocess import *
import argparse

USAGE = "USAGE: get_timestamp.py -f <sham output> -1 <value 1> -2 <value 2>\n"

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sham', type=str, help="output of sham.pl")
parser.add_argument('--x', type=float, help="X value from FEL")
parser.add_argument('--y', type=float, help="Y value from FEL")
parser.add_argument('--xtc', type=str, help="Location of the trajectory")
parser.add_argument('--tpr', type=str, help="Location for the structure file")
parser.add_argument('--prefix', type=str, help="Prefix for output pdb (including output directory!)")
args = parser.parse_args()

# Open sham output
x_values, y_values, time_values = [], [], []

with open(args.sham) as shamfile:
    for line in shamfile:
        if line[0] != "#" and len(line.split()) == 3:
            t,x,y = line.split()
            x_values.append(float(x))
            y_values.append(float(y))
            time_values.append(float(t))

def find_common_frame(min_x, min_y):
    for xval in min_x:
        xframe = xval[0]
        for yval in min_y:
            yframe = yval[0]
            if xframe == yframe:
                return (xframe, xval[1], yval[1])
    return (None, None, None)

# If you cannot find anything, try increasing the nval variable
nval = 500
min_x = sorted(enumerate(x_values), key=lambda x: abs(x[1]-args.x))[:nval]
min_y = sorted(enumerate(y_values), key=lambda x: abs(x[1]-args.y))[:nval]

frame, x, y = find_common_frame(min_x, min_y)

if not frame:
    print("No timestamp found..")
    sys.exit(0)

#Make the pdb!
command = "gmx trjconv -f {} -s {} -o {}_{}.pdb -b {} -e {}".format(args.xtc, args.tpr, args.prefix, time_values[frame], time_values[frame], time_values[frame])
p = Popen(command.split(), stdin=PIPE)
p.communicate(b'0\n')
p.wait()

print("## T = %s (%s, %s)" %(time_values[frame], x, y))