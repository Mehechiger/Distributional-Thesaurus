#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import re
import math
import dlib
import timeit
import numpy as np
import numexpr as ne
import pandas as pd
from tqdm import tqdm, trange


def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        # arg_str = ', '.join(repr(arg) for arg in args)
        # print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        print('[%0.8fs] %s()\n' % (elapsed, name))
        return result
    return clocked


class AzDist:
    def __init__(self, corpsDir, cats="A|ADV|N|V", wFunc="NONE", sFunc="COS"):
        self.corpsDir, self.wFunc, self.sFunc = corpsDir, wFunc, sFunc
        self.cats = {cat for cat in cats.split('|')}

        self.ctts = self.cttsExtract()
        self.wVecs = self.vecsWeight()
        self.thesGen()
        return

    def cttsExtract(self):
        def ExtractNGrm(corpDir, ctts, n=2):
            def hasNbs(s):
                return any(char.isdigit() for char in s)

            def isNbs(s):
                return all(char.isdigit() for char in s)

            def corpLoadLC_N(corpDir, n):
                with open(corpDir, 'r') as corpFile:
                    try:
                        corpus = re.compile(
                            '(?:\n)|(?:(?:[^\t]*\t[^\t]*\t)(.*?)\t(.*?)\t.*\n)').findall(corpFile.read())
                        return corpus
                    except:
                        print("Error: corpus file %s unusable, has passed." % corpDir)
                        return []

            corpus = corpLoadLC_N(corpDir, n)
            for i in trange(len(corpus)):
                lemmaI, catI = corpus[i]
                if catI in self.cats and not hasNbs(lemmaI):
                    try:
                        ctts[catI][lemmaI]['WORD_COUNT'] += 1
                    except KeyError:
                        ctts[catI][lemmaI] = {'WORD_COUNT': 1}
                    cttsLemmaI = ctts[catI][lemmaI]

                    for j in range(1, n):
                        if i-j == -1 or corpus[i-j] == ('', ''):
                            k = n-j+1
                            while k > 0:
                                prevGrm = '%s%s%s' % ('-', str(n-k+1), '^')
                                try:
                                    cttsLemmaI[prevGrm] += 1
                                except KeyError:
                                    cttsLemmaI[prevGrm] = 1
                                k -= 1
                            break
                        else:
                            tmp = "".join(corpus[i-j])
                            prevGrm = '(NBS)' if isNbs(corpus[i-j][0]) else tmp
                            prevGrm = '%s%s%s' % ('-', str(j), prevGrm)
                            try:
                                cttsLemmaI[prevGrm] += 1
                            except KeyError:
                                cttsLemmaI[prevGrm] = 1

                        if i+j == len(corpus) or corpus[i+j] == ('', ''):
                            k = n-j+1
                            while k > 0:
                                nextGrm = '%s%s%s' % ('+', str(n-k+1), '$')
                                try:
                                    cttsLemmaI[nextGrm] += 1
                                except KeyError:
                                    cttsLemmaI[nextGrm] = 1
                                k -= 1
                            break
                        else:
                            tmp = "".join(corpus[i+j])
                            nextGrm = '(NBS)' if isNbs(corpus[i+j][0]) else tmp
                            nextGrm = '%s%s%s' % ('+', str(j), nextGrm)
                            try:
                                cttsLemmaI[nextGrm] += 1
                            except KeyError:
                                cttsLemmaI[nextGrm] = 1
                    ctts[catI][lemmaI] = cttsLemmaI
            return ctts

        ctts = {}
        for cat in self.cats:
            ctts[cat] = {}
        for i in trange(len(self.corpsDir)):
            ctts = ExtractNGrm(corpsDir[i], ctts, 2)

        for cttsCATk, cttsCATv in tqdm(ctts.items()):
            res = {}
            for k, v in cttsCATv.items():
                if v['WORD_COUNT'] > 10:
                    tmp = {kk: vv for kk, vv in v.items() if kk !=
                           'WORD_COUNT' and vv > 5}
                    #tmp['WORD_COUNT'] = v['WORD_COUNT']
                    res[k] = tmp
            ctts[cttsCATk] = res
        return ctts

    def vecsWeight(self):
        return self.ctts

    """
    @clock
    def simsCal(self):
        print("Calculating words similarities...")
        sims = {}
        for wVecsCATk, wVecsCATv in self.wVecs.items():
            simsCAT, CaledPairs = {}, {}
            for wVec1k in tqdm(wVecsCATv.keys()):
                simsWord = {}
                wVec1 = wVecsCATv[wVec1k]
                for wVec2k in wVecsCATv.keys():
                    if wVec2k == wVec1k:
                        continue
                    try:
                        if CaledPairs[wVec2k] == wVec1k:
                            simsCAT[wVec1k][wVec2k] = simsCAT[wVec2k][wVec1k]
                    except KeyError:
                        pass
                    sumXiYi, sumXiPow, sumYiPow = 0, 0, 0
                    wVec2 = wVecsCATv[wVec2k]
                    for k1, v1 in wVec1.items():
                        if k1 != "WORD_COUNT":
                            sumXiPow += v1*v1
                            if k1 in wVec2:
                                sumXiYi += v1*wVec2[k1]
                    for k2, v2 in wVec2.items():
                        if k2 != "WORD_COUNT":
                            sumYiPow += v2*v2
                    try:
                        simsWord[wVec2k] = sumXiYi / \
                            (math.sqrt(sumXiPow*sumYiPow))
                    except:
                        simsWord[wVec2k] = 0
                    CaledPairs[wVec2k] = wVec1k
                simsCAT[wVec1k] = sorted(
                    simsWord.items(), key=lambda item: item[1], reverse=True)[:10]
            sims[wVecsCATk] = simsCAT
        return sims
    """

    @clock
    def simsCal(self):
        print("Calculating words similarities...")
        sims = {}
        for cat in self.cats:
            wVecsCatDf = pd.DataFrame(self.wVecs[cat])
            for i in tqdm(wVecsCatDf):
                t = wVecsCatDf[i]  # .fillna(0)
                n1 = ne.evaluate("sum(t**2)")
                for j in wVecsCatDf:
                    u = wVecsCatDf[j]  # .fillna(0)
                    n2 = ne.evaluate("sum(u**2)")
                    n3 = ne.evaluate("sum(t*u)")
                    n4 = n3/math.sqrt(n1*n2) if n1 != 0 and n2 != 0 else 0
        return sims

    def thesGen(self):
        return


if __name__ == "__main__":
    corpDirFolder = "/Users/mehec/nlp/prjTAL_L3/donnees/"
    corpsDir = ['%s%s' % (corpDirFolder, corpDir) for corpDir in os.listdir(
        corpDirFolder) if corpDir[-8:] == ".outmalt"]
    dist = AzDist(corpsDir[:1], "A")
    for key, value in dist.simsCal().items():
        pass
        # print(key, ' : ', value, '\n\n')
