#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Load the Pandas libararies with alias 'pd'
import pandas as pd
import numpy as np
import argparse as ap
import scipy
from scipy.stats.stats import pearsonr as pr
from scipy.stats.stats import spearmanr as sr
import matplotlib.pyplot as plt
import os

# evaluating the AzDist
arg_parser = ap.ArgumentParser()
arg_parser.add_argument("-sd", "--resdir", help="results directory")
arg_parser.add_argument("-fd", "--refdir", help="reference directory")
args = arg_parser.parse_args()

res_dataset_ori = pd.read_csv(args.resdir, index_col=0).fillna(float(10000))
res_dataset_ori[res_dataset_ori < -10000] = float(10000)
res_dataset_ori[res_dataset_ori > 10000] = float(10000)
res_dataset_ori[res_dataset_ori == float(0)] = float(10000)

res_dataset = []
ref_dataset = []
word_pairs = []
with open(args.refdir, 'r') as file:
    while(True):
        line = file.readline()
        if line == "":
            break
        line = line.split()
        try:
            res_dataset.append(res_dataset_ori[line[0]][line[1]])
            if res_dataset[-1] == float(10000):
                res_dataset.pop()
                continue
            ref_dataset.append(float(line[2]))
            word_pairs.append(line[0]+"/"+line[1])
        except KeyError:
            try:
                res_dataset.append(res_dataset_ori[line[1]][line[0]])
                if res_dataset[-1] == float(10000):
                    res_dataset.pop()
                    continue
                ref_dataset.append(float(line[2]))
                word_pairs.append(line[1]+"/"+line[0])
            except KeyError:
                continue

print(ref_dataset)
print(res_dataset)
print(len(ref_dataset))

plt.scatter(ref_dataset, res_dataset)
plt.xlabel("Humain scores")
plt.ylabel("Distributionnal scores")
plt.grid()

fp2 = np.polyfit(ref_dataset, res_dataset, 1)
f2 = np.poly1d(fp2)
fx = np.linspace(0, ref_dataset, 1000)
plt.plot(fx, f2(fx), linewidth=2, color='g')

plt.show()

resPr = pr(ref_dataset, res_dataset)
resSr = sr(ref_dataset, res_dataset)
print(resPr)
print(resSr)
