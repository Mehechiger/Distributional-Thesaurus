#!/usr/bin/python3
# -*- coding:utf-8 -*-

# precise devision regardless of number type (res all in float)
from __future__ import division
import os
import re
import math
import timeit
# progress bar, wrap it around an iterator and it'll work
from tqdm import tqdm, trange


# function timer, as decorator
# copied codes, normally no need to modify
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
    # arguments: corpora file directory, categories of words to analyse
    #           choice of weighting function, choice of similarity function
    def __init__(self, corpsDir, cats="A|ADV|N|V", wFunc="NONE", sFunc="COS"):
        # assignment of attributs
        self.corpsDir, self.wFunc, self.sFunc = corpsDir, wFunc, sFunc
        self.cats = {cat for cat in cats.split('|')}

        # the program's working order:
        # extract contexts into word vectors
        # > weight the word vectors
        # > calculate word similairities
        self.ctts = self.cttsExtract()
        self.wVecs = self.vecsWeight()
        self.thesGen()
        return

    # extract word contexts from given corpus(corpora) into word vectors
    # output format:
    # {"category1":
    #               {"word1":
    #                           {"context1":Nb,
    #                           "context2":Nb,
    #                           ...
    #                           },
    #               "word2":
    #                           {...
    #                           },
    #               ...
    #               },
    # category2:
    #               {...
    #               },
    # ...
    # }
    def cttsExtract(self):
        # extract in n-gram format
        def ExtractNGrm(corpDir, ctts, n=2):
            # determine whether a string has numbers in it
            def hasNbs(s):
                return any(char.isdigit() for char in s)

            # determine whether a string is composed entirely of numbers
            def isNbs(s):
                return all(char.isdigit() for char in s)

            # load a corpus file while trimming off all useless data
            # output format:
            # [["this","D"],
            #  ["be", "V"],
            #  ["a", "D"],
            #  ["exemple", "N"],
            #  [".", "PONCT"],
            #  ["", ""],
            #  ["that", "D"],
            #  ...
            # ]
            # ["",""] being the representation of an empty line,
            # which is the seperator of sentences
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
            for i in trange(len(corpus)):  # go over the corpus with a loop for just once
                lemmaI, catI = corpus[i]
                # we only need to check words of the categories which we're insterested in
                # and we don't want words with numbers
                if catI in self.cats and not hasNbs(lemmaI):
                    # increment the word count if it is already in the inventory
                    # else create one
                    # the try/except trick runs faster with many repeats, especially when the dict becames larger
                    try:
                        ctts[catI][lemmaI]['WORD_COUNT'] += 1
                    except KeyError:
                        ctts[catI][lemmaI] = {'WORD_COUNT': 1}
                    # create a local variable, it is said to be able to boost up the speed
                    # and it makes the code clearer
                    cttsLemmaI = ctts[catI][lemmaI]

                    passPrev, passNext = False, False
                    # j is the position we need to extract the context
                    # its value varies within certain range: [-(n-1),0) and (0, (n-1)]
                    for j in range(1, n):
                        # this part deals with all previous positions

                        if passPrev == True:
                            pass
                        # make sure we're not running into the beginning of a sentence (or the corpus)
                        # if we are...
                        elif i-j == -1 or corpus[i-j] == ('', ''):
                            # then just add contexts of this type: "-3^", "-1^", ...
                            # 1 while loop would suffice
                            k = n-j+1
                            while k > 0:
                                prevGrm = '%s%s%s' % ('-', str(n-k+1), '^')
                                try:
                                    cttsLemmaI[prevGrm] += 1
                                except KeyError:
                                    cttsLemmaI[prevGrm] = 1
                                k -= 1
                            passPrev = True  # everything's done, we can pass the prevGrms next time
                        # if not, add contexts of this type: "-5pommeN", "-1leD", ...
                        # for contexts that are entirely numbers: "-2(NBS)", "-3(NBS)", ...
                        # for contexts that are ponctuation marks: "-2(PONCT)", "-4(PONCT)", ...
                        else:
                            tmp = "".join(corpus[i-j])
                            if isNbs(corpus[i-j][0]):
                                prevGrm = '(NBS)'
                            elif corpus[i-j][1] == 'PONCT':
                                prevGrm = '(PONCT)'
                            else:
                                prevGrm = tmp
                            prevGrm = '%s%s%s' % ('-', str(j), prevGrm)
                            try:
                                cttsLemmaI[prevGrm] += 1
                            except KeyError:
                                cttsLemmaI[prevGrm] = 1

                        # this part deals with all following positions, basically same as above
                        if passNext == True:
                            pass
                        elif i+j == len(corpus) or corpus[i+j] == ('', ''):
                            k = n-j+1
                            while k > 0:
                                nextGrm = '%s%s%s' % ('+', str(n-k+1), '$')
                                try:
                                    cttsLemmaI[nextGrm] += 1
                                except KeyError:
                                    cttsLemmaI[nextGrm] = 1
                                k -= 1
                            passNext = True
                        else:
                            tmp = "".join(corpus[i+j])
                            if isNbs(corpus[i+j][0]):
                                nextGrm = '(NBS)'
                            elif corpus[i+j][1] == 'PONCT':
                                nextGrm = '(PONCT)'
                            else:
                                nextGrm = tmp
                            nextGrm = '%s%s%s' % ('+', str(j), nextGrm)
                            try:
                                cttsLemmaI[nextGrm] += 1
                            except KeyError:
                                cttsLemmaI[nextGrm] = 1

                    # update the main dict after each loop
                    ctts[catI][lemmaI] = cttsLemmaI
            return ctts

        # initialize the main dict, creating dicts of each category needed
        ctts = {}
        for cat in self.cats:
            ctts[cat] = {}
        # go over the corpora file one by one
        # (only n-gram mode available for now. To be extended.)
        for i in trange(len(self.corpsDir)):
            # haven't decide yet how to pass in the n parameter of n-gram
            # so just put it here for now, by default bigram
            ctts = ExtractNGrm(corpsDir[i], ctts, 3)

        # this part trims off words and contexts that occur too rarely
        # it is said to be able to boost up the speed and increase accuracy
        for cttsCATk, cttsCATv in tqdm(ctts.items()):
            # functions by creating a new main dict and copy everything relavant
            res = {}
            for k, v in cttsCATv.items():
                if v['WORD_COUNT'] > len(self.corpsDir):  # word count threshold
                    res[k] = {kk: vv for kk, vv in v.items() if kk !=
                              'WORD_COUNT' and vv > math.ceil(len(self.corpsDir)/10)}  # context count threshold
            ctts[cttsCATk] = res
        return ctts

    # to be written
    def vecsWeight(self):
        return self.ctts

    @clock
    # output format:
    # {"category1":
    #               {"word1":
    #                           {"word2":simScore,
    #                           "word3":simScore,
    #                           "word4":simScore,
    #                           ...
    #                           },
    #               "word2":
    #                           {"word1":simScore,
    #                           "word3":simScore,
    #                           "word4":simScore,
    #                           ...
    #                           },
    #               ...
    #               },
    # category2:
    #               {...
    #               },
    # ...
    # }
    def simsCal(self):
        print("Calculating words similarities...")
        sims = {}
        for wVecsCATk, wVecsCATv in self.wVecs.items():  # k, v for key and value
            # normalize all vectors before calculation of sims
            for wVeck, wVecv in wVecsCATv.items():
                try:
                    maxV = max(wVecv.values())
                    wVecsCATv[wVeck] = {k: v/maxV for k,
                                        v in wVecv.items() if v != 0 and v != maxV}  # unit vectors can cause great amount of anormalies and have to be eliminated
                except:
                    wVecsCATv[wVeck] = {}

            # CaledPairs stocks all pairs already calculated
            simsCAT, CaledPairs = {}, {}
            # 1st loop of all words, (wVec1k: 1 for 1st word in a pair)
            for wVec1k, wVec1v in tqdm(wVecsCATv.items()):
                simsWord = {}  # stocks the simScores of this word comparing to all other words
                # sumXiPow for sum of the square of each component of a vector x
                for wVec2k in wVecsCATv.keys():  # 2 for 2nd word in a pair
                    if wVec2k == wVec1k:  # pass if they're the same word
                        continue
                    try:  # again, the try/except trick
                        # if the pair is already calculated (previously, in a different order)
                        # then just copy the stocked value
                        if CaledPairs[wVec2k] == wVec1k:
                            simsCAT[wVec1k][wVec2k] = simsCAT[wVec2k][wVec1k]
                    except KeyError:
                        pass
                    # initialize 2 variables, where
                    # sumXiYi for dot product of 2 vectors x and y
                    sumXiYi = 0
                    wVec2v = wVecsCATv[wVec2k]
                    # only if we begin a new outer loop, do we need to recalculate sumXiPow
                    isNewOuterLoop = 0
                    for k1, v1 in wVec1v.items():
                        # if the same dimension(context) exist in word2 then calculate sumXiYi
                        # do not use try/except because the data are sparse (rarely have the same context)
                        if k1 in wVec2v:
                            sumXiYi += v1*wVec2v[k1]
                    # no need to continue if the numerator is 0
                    if sumXiYi == 0:
                        CaledPairs[wVec2k] = wVec1k
                        continue
                    # cosine equation: already normalized, so just the numerator part
                    simsWord[wVec2k] = sumXiYi
                    """
                    if simsWord[wVec2k] == 1:
                        print(wVec1k, wVec1v)
                        print(wVec2k, wVec2v)
                        print("\n\n\n")
                    """
                    # stock each calculed pair
                    CaledPairs[wVec2k] = wVec1k
                # keep only the first 10 words with highest simScores
                # to be seen wether or not it is better to put in thesGen part
                simsCAT[wVec1k] = sorted(
                    simsWord.items(), key=lambda item: item[1], reverse=True)[:10]
            sims[wVecsCATk] = simsCAT
        return sims

    def thesGen(self):
        return


# calling and testing
if __name__ == "__main__":
    # modify to your corpora directory
    # to be made a parameter for commandline access
    corpDirFolder = "/Users/mehec/nlp/prjTAL_L3/donnees/"

    # make sure we only take file with filename extenstion ".outmalt"
    # keep this line
    corpsDir = ['%s%s' % (corpDirFolder, corpDir) for corpDir in os.listdir(
        corpDirFolder) if corpDir[-8:] == ".outmalt"]

    # instantiate the class
    # for fast testing, only take minimal amount of corpora and categories
    dist = AzDist(corpsDir[:5], "A")
    for key, value in dist.simsCal().items():
        # print all similarities results to be examined
        print(key, ' : ', value, '\n\n')
