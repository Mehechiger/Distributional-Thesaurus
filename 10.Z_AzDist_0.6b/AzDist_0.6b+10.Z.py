#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import os
import time
import timeit
# progress bar, just wrap it around an iterator
from tqdm import tqdm, trange
import re


# timer decorator
def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
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
                lcI = lemmaI+catI
                if catI in self.cats and not hasNbs(lemmaI):
                    ctts[catI][lemmaI] = ctts[catI].get(
                        lemmaI, {'WORD_COUNT': 0})
                    ctts[catI][lemmaI]['WORD_COUNT'] += 1

                    for j in range(1, n):
                        if i-j == -1 or corpus[i-j] == ('', ''):
                            k = n-j+1
                            while k > 0:
                                prevGrm = '-'+str(n-k+1)+'^'
                                ctts[catI][lemmaI][prevGrm] = ctts[catI][lemmaI].get(
                                    prevGrm, 0)+1
                                k -= 1
                            break
                        else:
                            tmp = "".join(corpus[i-j])
                            prevGrm = '(NBS)' if isNbs(corpus[i-j][0]) else tmp
                            prevGrm = '-'+str(j)+prevGrm
                            ctts[catI][lemmaI][prevGrm] = ctts[catI][lemmaI].get(
                                prevGrm, 0)+1
                    for j in range(1, n):
                        if i+j == len(corpus) or corpus[i+j] == ('', ''):
                            k = n-j+1
                            while k > 0:
                                nextGrm = '+'+str(n-k+1)+'$'
                                ctts[catI][lemmaI][nextGrm] = ctts[catI][lemmaI].get(
                                    nextGrm, 0)+1
                                k -= 1
                            break
                        else:
                            tmp = "".join(corpus[i+j])
                            nextGrm = '(NBS)' if isNbs(corpus[i+j][0]) else tmp
                            nextGrm = '+'+str(j)+nextGrm
                            ctts[catI][lemmaI][nextGrm] = ctts[catI][lemmaI].get(
                                nextGrm, 0)+1
            return ctts
        ctts = {}
        for cat in self.cats:
            ctts[cat] = {}
        # for corpDir in self.corpsDir:
        #    ctts = ExtractNGrm(corpDir, ctts, 2)
        for i in trange(len(self.corpsDir)):
            ctts = ExtractNGrm(corpsDir[i], ctts, 2)

        for cttsCATk, cttsCATv in ctts.items():
            res = {}
            for k, v in cttsCATv.items():
                tmp = {kk: vv for kk, vv in v.items() if kk !=
                       'WORD_COUNT' and vv > 10}
                if len(tmp.keys()) > 5:
                    tmp['WORD_COUNT'] = v['WORD_COUNT']
                    res[k] = tmp
            ctts[cttsCATk] = res
        return ctts

        """
        @clock
        def ParseSents(sentences):
            def cttsMerge(ctts1, ctts2):
                for k2, v2 in ctts2.items():
                    if k2 in ctts1.keys():
                        for k2k, v2v in v2.items():
                            if k2k in ctts1[k2].keys():
                                ctts1[k2][k2k] = ctts1[k2].get(k2k, 0)+v2v
                            else:
                                ctts1[k2][k2k] = v2v
                    else:
                        ctts1[k2] = v2
                return ctts1

            print("Parsing sentences...")
            cttsSents = {}
            for sentence in sentences:
                cttsMerge(cttsSents, ParseSent(sentence))

            res = {}
            for k, v in cttsSents.items():
                tmp = {kk: vv for kk, vv in v.items(
                ) if kk != 'WORD_COUNT' and vv > 10}
                if len(tmp.keys()) > 5:
                    tmp['WORD_COUNT'] = v['WORD_COUNT']
                    res[k] = tmp
            return res

        sentences = []
        for corpDir in tqdm(self.corpsDir):
            sentences.extend(sentsSplit(ReadFile(corpDir)))
        return ParseSents(sentences)
    """

    def vecsWeight(self):
        return self.ctts

    @clock
    def simsCal(self):
        print("Calculating words similarities...")
        sims = {}
        for wVecsCATk, wVecsCATv in self.wVecs.items():
            simsCAT = {}
            CaledPairs = {}
            for wVec1k in tqdm(wVecsCATv.keys()):
                simsWord = {}
                wVec1 = wVecsCATv[wVec1k]
                for wVec2k in wVecsCATv.keys():
                    if wVec2k == wVec1k:
                        continue
                    elif wVec2k in CaledPairs:
                        if CaledPairs[wVec2k] == wVec1k:
                            simsCAT[wVec1k][wVec2k] = simsCAT[wVec2k][wVec1k]
                    sumXiYi = 0.0
                    sumXiPow = sumYiPow = 0.0
                    wVec2 = wVecsCATv[wVec2k]
                    for k1, v1 in wVec1.items():
                        if k1 != "WORD_COUNT":
                            if v1 != 0:
                                sumXiPow += v1*v1
                            if k1 in wVec2:
                                sumXiYi += v1*wVec2[k1]
                    for k2, v2 in wVec2.items():
                        if k2 != "WORD_COUNT" and v2 != 0:
                            sumYiPow += v2*v2
                    if sumXiPow != 0 and sumYiPow != 0 and sumXiYi != 0:
                        simsWord[wVec2k] = sumXiYi / \
                            (math.sqrt(sumXiPow)*math.sqrt(sumYiPow))
                    CaledPairs[wVec2k] = wVec1k
                if simsWord == {}:
                    continue
                simsCAT[wVec1k] = sorted(
                    simsWord.items(), key=lambda item: item[1], reverse=True)[:10]
            sims[wVecsCATk] = simsCAT

        """
        sims = {}
        CaledPairs = {}
        for wVec1k in tqdm(self.wVecs.keys()):
            simsWord = {}
            wVec1 = self.wVecs[wVec1k]
            for wVec2k in tqdm(self.wVecs.keys()):
                if wVec2k == wVec1k:
                    continue
                elif wVec2k in CaledPairs.keys():
                    if CaledPairs[wVec2k] == wVec1k:
                        sims[wVec1k][wVec2k] = sims[wVec2k][wVec1k]
                sumXiYi = 0.0
                sumXiPow = sumYiPow = 0.0
                wVec2 = self.wVecs[wVec2k]
                if wVec2k.split("_")[-1] == wVec1k.split("_")[-1]:
                    for k1, v1 in wVec1.items():
                        if k1 != "WORD_COUNT":
                            if v1 != 0:
                                sumXiPow += v1*v1
                            if k1 in wVec2.keys():
                                sumXiYi += v1*wVec2[k1]
                    for k2, v2 in wVec2.items():
                        if k2 != "WORD_COUNT" and v2 != 0:
                            sumYiPow += v2*v2
                    if sumXiPow != 0 and sumYiPow != 0 and sumXiYi != 0:
                        simsWord[wVec2k] = sumXiYi / \
                            (math.sqrt(sumXiPow)*math.sqrt(sumYiPow))
                CaledPairs[wVec2k] = wVec1k
            if simsWord == {}:
                continue
            sims[wVec1k] = sorted(
                simsWord.items(), key=lambda item: item[1], reverse=True)[:10]
            """
        return sims

    def thesGen(self):
        return


if __name__ == "__main__":
    corpDirFolder = "/Users/mehec/nlp/prjTAL_L3/donnees/"
    corpsDir = [corpDirFolder+corpDir for corpDir in os.listdir(
        corpDirFolder) if corpDir[-8:] == ".outmalt"]
    dist = AzDist(corpsDir)
    for key, value in dist.simsCal().items():
        pass
        #print(key, ' : ', value, '\n\n')
