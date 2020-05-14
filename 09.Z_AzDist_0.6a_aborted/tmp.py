#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import os
import time
import timeit
# progress bar, just wrap it around an iterator
from tqdm import tqdm, trange


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
    # def __init__(self, corpsDir, cols="2|3", cats="A|ADV|N|V", wFunc="NONE", sFunc="COS"):
    def __init__(self, corpsDir, cats="A|ADV|N|V", wFunc="NONE", sFunc="COS"):
        self.corpsDir, self.wFunc, self.sFunc = corpsDir, wFunc, sFunc
        # self.cols = [int(col) for col in cols.split('|')]
        self.cats = [cat for cat in cats.split('|')]

        self.ctts = self.cttsExtract()
        self.wVecs = self.vecsWeight()
        self.thesGen()
        return

    def cttsExtract(self):
        @clock
        def ExtractNGrm(corpDir, n=2):
            def hasNbs(s):
                return any(char.isdigit() for char in s)

            def corpLoadLC(corpDir, n):
                with open(corpDir, 'r') as corpFile:
                    try:
                        #corpus = []
                        corpus = {}
                        for i in range(n-1):
                            #corpus.append(['', '^'])
                            corpus[i] = ['', '^']
                        count = 0
                        for line in corpFile.readlines():
                            if line == "\n":
                                for i in range(n-1):
                                    #corpus.append(['', '$'])
                                    corpus[n-1+count+i] = ['', '^']
                                for i in range(n-1):
                                    #corpus.append(['', '^'])
                                    corpus[n-1+count+n-1+i] = ['', '^']
                            else:
                                # corpus.append([line.split("\t")[col]
                                #               for col in [2, 3]])
                                corpus[n-1+count] = [line.split("\t")[col]
                                                     for col in [2, 3]]
                            count += 1
                        return corpus
                    except:
                        print("Error: corpus file %s unusable, has passed." % corpDir)
                        return []

            corpus = corpLoadLC(corpDir, n)
            for cat in self.cats:
                locals()["ctts"+cat] = {}
            # for i in trange(len(corpus)):
            for i in trange(len(corpus.keys())):
                [lemmaI, catI] = corpus[i]
                lcI = lemmaI+catI
                if catI in self.cats and not hasNbs(lemmaI):
                    VARcttsCAT = 'ctts'+catI
                    exec("%s[lemmaI]=%s.get(lemmaI, {'WORD_COUNT':0})" % (
                        VARcttsCAT, VARcttsCAT))
                    exec("%s[lemmaI]['WORD_COUNT']+=1" % VARcttsCAT)
                    #prevGrms = "".join(sum(corpus[i-n+1:i-1], []))+"<"
                    #nextGrms = ">"+"".join(sum(corpus[i+1:i+n-1], []))
                    tmp = ""
                    for j in range(i-n+1, i):
                        [a, b] = corpus[j]
                        tmp = tmp+a+b
                    prevGrms = tmp+'<'
                    tmp = ""
                    for j in range(i+1, i+n):
                        [a, b] = corpus[j]
                        tmp = tmp+a+b
                    nextGrms = '>'+tmp
                    exec("%s[lemmaI][prevGrms]=%s[lemmaI].get(prevGrms, 0)+1" %
                         (VARcttsCAT, VARcttsCAT))
                    exec("%s[lemmaI][nextGrms]=%s[lemmaI].get(nextGrms, 0)+1" %
                         (VARcttsCAT, VARcttsCAT))
            ctts = {}
            for cat in self.cats:
                exec("ctts[cat]=%s" % 'ctts'+cat)
            return ctts
        return ExtractNGrm(corpsDir[2])

        """
        @clock
        def ParseSents(sentences):
            def ParseSent(sentence):
                def hasNbs(s):
                    return any(char.isdigit() for char in s)

                prevLemma__cat = ""
                cttsSent = {}
                for i in range(len(sentence)):
                    lemmaI, catI = sentence[i][0], sentence[i][1]
                    lemma__catI = lemmaI+"__"+catI
                    if catI in self.cats and not hasNbs(lemmaI):
                        cttsSent[lemma__catI] = cttsSent.get(
                            lemma__catI, {'WORD_COUNT': 1})
                        if i > 0:
                            cttsSent[lemma__catI][(-1, prevLemma__cat)
                                                  ] = cttsSent[lemma__catI].get((-1, prevLemma__cat), 0)+1
                        else:
                            cttsSent[lemma__catI][(-1, "SENT_BEG")
                                                  ] = cttsSent[lemma__catI].get((-1, "SENT_BEG"), 0)+1
                        if i+1 < len(sentence):
                            nextLemma__cat = sentence[i +
                                                      1][0]+"__"+sentence[i+1][1]
                            cttsSent[lemma__catI][(1, nextLemma__cat)
                                                  ] = cttsSent[lemma__catI].get((1, nextLemma__cat), 0)+1
                        else:
                            cttsSent[lemma__catI][(1, "SENT_END")] = cttsSent[lemma__catI].get(
                                (1, "SENT_END"), 0)+1
                    prevLemma__cat = lemma__catI
                return cttsSent

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
        return sims

    def thesGen(self):
        return


if __name__ == "__main__":
    corpDirFolder = "/Users/mehec/nlp/prjTAL_L3/donnees/"
    corpsDir = [corpDirFolder+corpDir for corpDir in os.listdir(
        corpDirFolder) if corpDir[-8:] == ".outmalt"]
    dist = AzDist(corpsDir[:5])
"""
    for key, value in dist.simsCal().items():
        print(key, ' : ', value, '\n\n')
"""
