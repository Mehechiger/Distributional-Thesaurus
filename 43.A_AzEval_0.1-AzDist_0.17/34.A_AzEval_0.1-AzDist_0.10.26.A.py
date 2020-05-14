#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Load the Pandas libararies with alias 'pd'
import pandas as pd
import argparse as ap
import scipy
from scipy.stats.stats import pearsonr as pr
from scipy.stats.stats import spearmanr as sr
import matplotlib.pyplot as plt
import os

# evaluating the AzDist
arg_parser = ap.ArgumentParser()
arg_parser.add_argument("-rd", "--resdir", help="results directory")
arg_parser.add_argument("-fd", "--refdir", help="reference directory")
args = arg_parser.parse_args()

resDir = ""
resDir = resDir+"/" if resDir[-1:] != "/" else resDir
resDir = ['%s' % resDir for resDir in os.listdir(args.resdir) if resDir[-4:] == ".csv"]

res_dataset = {0:{0:0}}
#res_dataset = pd.DataFrame(res_dataset)
#res_dataset = pd.concat(resDir)
frames = ['%' i, % res_dataset=pd.DataFrame(pd.read_csv(res)) for i in range(len(resDir))]

for i in range(len(resDir)):
    res_dataset = pd.DataFrame(pd.read_csv(resDir[i]))
    res_dataset = pd.concat(res_dataset)
    #res_dataset = pd.concat([res_dataset, pd.read_csv(resDir[i])], sort=True)

ref_dataset = {}
with open(args.refdir, "r") as file:
    while(True):
        ligne = file.readline()
        if ligne == "":
            break
        ligne = ligne.split()
        try:
            ref_dataset[ligne[0]][ligne[1]]=ligne[2]
        except KeyError:
            ref_dataset[ligne[0]]={}

ref_dataset = pd.DataFrame(ref_dataset)

lIndex=[]
for index in ref_dataset.index:
    try:
        res_dataset.loc[index,:]
        lIndex.append(index)
    except KeyError:
        pass
lColumns=[]
for column in ref_dataset.columns:
    try:
        res_dataset.loc[:,column]
        lColumns.append(column)
    except KeyError:
        pass

#res_dataset = res_dataset[ref_dataset.index, ref_dataset.columns]
res_dataset = res_dataset.loc[lIndex, lColumns]
print(ref_dataset)
print(res_dataset)    

    

      
    
    
    
    
