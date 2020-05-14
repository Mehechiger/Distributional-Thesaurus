#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import os
import time
import timeit


def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        #print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        print('[%0.8fs] %s(args)' % (elapsed, name))
        return result
    return clocked


class AzDist:
    def __init__(self, corpsDir, cols="2|3", cats="A|ADV|N|V", wFunc="NONE", sFunc="COS"):
        self.corpsDir, self.wFunc, self.sFunc = corpsDir, wFunc, sFunc
        self.cols = [int(col) for col in cols.split('|')]
        self.cats = [cat for cat in cats.split('|')]

        self.ctts = self.cttsParse()
        self.wVecs = self.vecsWeight()
        self.thesGen()
        return

    def cttsParse(self):
        @clock
        def ReadFile():
            corpora = []
            for corpDir in self.corpsDir:
                with open(corpDir, 'r') as corpus:
                    try:
                        corpora.extend(corpus.readlines())
                    except:
                        pass
            return corpora

        def sentsSplit(corpora):
            sentences = []
            sentence = []
            for line in corpora:
                if line == '\n':
                    sentences.append(sentence)
                    sentence = []
                else:
                    word = [line.split("\t")[col] for col in self.cols]
                    sentence.append(word)
            return sentences

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

            cttsSents = {}
            for sentence in sentences:
                cttsMerge(cttsSents, ParseSent(sentence))

            res = {}
            for k, v in cttsSents.items():
                tmp = {kk: vv for kk, vv in v.items(
                ) if kk != 'WORD_COUNT' and vv > 0}
                if len(tmp.keys()) > 0:
                    tmp['WORD_COUNT'] = v['WORD_COUNT']
                    res[k] = tmp
            return res

        sentences = sentsSplit(ReadFile())
        return ParseSents(sentences)

    def vecsWeight(self):
        return self.ctts

    @clock
    def simsCal(self):
        sims = {}
        count1 = 0
        for wVec1k in self.wVecs.keys():
            if count1 % 500 == 0:
                print("\rProcessing %s/%s primary words" %
                      (count1, len(self.wVecs.keys())))
            count1 += 1
            simsWord = {}
            wVec1 = self.wVecs[wVec1k]
            for wVec2k in self.wVecs.keys():
                if wVec2k == wVec1k:
                    continue
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
            if simsWord == {}:
                continue
            sims[wVec1k] = sorted(
                simsWord.items(), key=lambda item: item[1], reverse=True)[:10]
        return sims

    def thesGen(self):
        return


if __name__ == "__main__":
    #dist = AzDist("estrepublicain.extrait-aa.19998.outmalt", cats="A")
    # dist = AzDist("EP.tcs.melt.utf8.split-ag.outmalt", "EP.tcs.melt.utf8.split-ad.outmalt",
    #              "EP.tcs.melt.utf8.split-bb.outmalt", "EP.tcs.melt.utf8.split-cx.outmalt", cats="A")
    corpDirFolder = "/Users/mehec/nlp/prjTAL_L3/donnees/"
    #corpDirFolder = "/Users/mehec/nlp/prjTAL_L3/codes/AzDist_0.3.1/"
    corpsDir = [corpDirFolder+corpDir for corpDir in os.listdir(
        corpDirFolder) if corpDir[-8:] == ".outmalt"]
    dist = AzDist(corpsDir[:10], cats="A")

    for key, value in dist.simsCal().items():
        print(key, ' : ', value, '\n\n')
