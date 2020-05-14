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

res_dataset = pd.read_csv(args.resdir, index_col=0)

ref_dataset = {}
with open(args.refdir, "r") as file:
    while(True):
        ligne = file.readline()
        if ligne == "":
            break
        ligne = ligne.split()
        try:
            ref_dataset[ligne[0]][ligne[1]] = float(ligne[2])
        except KeyError:
            ref_dataset[ligne[0]] = {}
            ref_dataset[ligne[0]][ligne[1]] = float(ligne[2])
        try:
            ref_dataset[ligne[1]][ligne[0]] = float(10000)
        except KeyError:
            ref_dataset[ligne[1]] = {}
            ref_dataset[ligne[1]][ligne[0]] = float(10000)

ref_dataset = pd.DataFrame(ref_dataset)

lIndex = []
lColumns = []
for index in ref_dataset.index:
    if index in res_dataset.index:
        lIndex.append(index)
    if index in res_dataset.columns:
        lColumns.append(index)
for column in ref_dataset.columns:
    if column in res_dataset.columns:
        lColumns.append(column)
    if column in res_dataset.index:
        lIndex.append(column)

ref_dataset = ref_dataset.loc[lIndex, lColumns].fillna(float(10000))
res_dataset = res_dataset.loc[lIndex, lColumns].fillna(float(10000))
res_dataset[res_dataset < -10000] = float(10000)
res_dataset[res_dataset > 10000] = float(10000)
res_dataset[res_dataset == float(0)] = float(10000)

print(ref_dataset)
print(res_dataset)

ref1DArr = np.ravel(ref_dataset)
res1DArr = np.ravel(res_dataset)

ref1DArr[np.argwhere(res1DArr == float(10000))] = float(10000)
res1DArr[np.argwhere(ref1DArr == float(10000))] = float(10000)

ref1DArr = np.delete(ref1DArr, np.argwhere(ref1DArr == float(10000)))
res1DArr = np.delete(res1DArr, np.argwhere(res1DArr == float(10000)))

print(ref1DArr)
print(res1DArr)

plt.scatter(ref1DArr, res1DArr)
plt.xlabel("Distributionnal scores")
plt.ylabel("Humain scores")
plt.autoscale(tight=True)
plt.grid()

fp2 = np.polyfit(ref1DArr, res1DArr, 1)
f2 = np.poly1d(fp2)
fx = np.linspace(0, ref1DArr, 1000)
plt.plot(fx, f2(fx), linewidth=2, color='g')

plt.show()

resPr = pr(ref1DArr, res1DArr)
resSr = sr(ref1DArr, res1DArr)
print(resPr)
print(resSr)
