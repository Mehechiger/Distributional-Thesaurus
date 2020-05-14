#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math


class AzDist:
    def __init__(self, corpDir, cols="2|3", cats="A|ADV|N|V", wFunc="NONE", sFunc="COS"):
        self.corpDir, self.wFunc, self.sFunc = corpDir, wFunc, sFunc
        self.cols = [int(col) for col in cols.split('|')]
        self.cats = [cat for cat in cats.split('|')]

        self.ctts = self.cttsParse()
        self.wVecs = self.vecsWeight()
        self.thesGen()
        return

    def cttsParse(self):
        def ReadFile():
            corpora = []
            for corpDir in self.corpDir:
                with open(corpDir, 'r') as corpus:
                    try:
                        corpora.extend(corpus.readlines())
                    except:
                        pass
            return corpora

        def sentsSplit(corpus):
            sentences = []
            sentence = []
            for line in corpus:
                if line == '\n':
                    sentences.append(sentence)
                    sentence = []
                else:
                    word = [line.split("\t")[col] for col in self.cols]
                    sentence.append(word)
            return sentences

        def ParseSents(sentences):
            def ParseSent(sentence):
                def hasNbs(s):
                    return any(char.isdigit() for char in s)
                prevLemma = ""
                cttsSent = {}
                for i in range(len(sentence)):
                    lemmaI, catI = sentence[i][0], sentence[i][1]
                    if catI in self.cats and not hasNbs(lemmaI):
                        cttsSent[lemmaI] = cttsSent.get(
                            lemmaI, {'WORD_COUNT': 1, 'WORD_CAT': catI})
                        if prevLemma != "":
                            cttsSent[lemmaI][(-1, prevLemma)
                                             ] = cttsSent[lemmaI].get((-1, prevLemma), 0)+1
                        if i+1 < len(sentence):
                            nextLemma = sentence[i+1][0]
                            cttsSent[lemmaI][(1, nextLemma)
                                             ] = cttsSent[lemmaI].get((1, nextLemma), 0)+1
                    prevLemma = lemmaI
                return cttsSent

            def cttsMerge(ctts1, ctts2):
                for k2, v2 in ctts2.items():
                    if k2 in ctts1.keys() and v2['WORD_CAT'] == ctts1[k2]['WORD_CAT']:
                        for k2k, v2v in v2.items():
                            if k2k != 'WORD_CAT' and k2k in ctts1[k2].keys():
                                ctts1[k2][k2k] = ctts1[k2].get(k2k, 0)+v2v
                    else:
                        ctts1[k2] = v2
                return ctts1

            cttsSents = {}
            for sentence in sentences:
                cttsMerge(cttsSents, ParseSent(sentence))
            res = {}
            for k, v in cttsSents.items():
                tmp = {kk: vv for kk, vv in v.items(
                ) if kk != 'WORD_COUNT' and kk != 'WORD_CAT' and vv > 0}
                if len(tmp.keys()) > 0:
                    tmp['WORD_COUNT'] = v['WORD_COUNT']
                    tmp['WORD_CAT'] = v['WORD_CAT']
                    res[k] = tmp
            return res

        sentences = sentsSplit(ReadFile())
        return ParseSents(sentences)

    def vecsWeight(self):
        return self.ctts

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
                if wVec2['WORD_CAT'] == wVec1['WORD_CAT']:
                    for k1, v1 in wVec1.items():
                        if k1 != "WORD_COUNT" and k1 != "WORD_CAT":
                            if v1 != 0:
                                sumXiPow += v1*v1
                            if k1 in wVec2.keys():
                                sumXiYi += v1*wVec2[k1]
                    for k2, v2 in wVec2.items():
                        if k2 != "WORD_COUNT" and k2 != "WORD_CAT" and v2 != 0:
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
    # dist = AzDist("estrepublicain.extrait-aa.19998.outmalt", cats="A")
    # dist = AzDist("EP.tcs.melt.utf8.split-ag.outmalt", "EP.tcs.melt.utf8.split-ad.outmalt",
    #              "EP.tcs.melt.utf8.split-bb.outmalt", "EP.tcs.melt.utf8.split-cx.outmalt", cats="A")
    import os
    corpDirFolder = "/Users/mehec/nlp/prjTAL_L3/donnees/"
    corpsDir = [corpDirFolder+corpDir for corpDir in os.listdir(corpDirFolder)]
    dist = AzDist(corpsDir[:5], cats="A")
    print(len(dist.ctts))
    """
    for key, value in dist.simsCal().items():
        print(key, ' : ', value, '\n\n')
    """
