#!/usr/bin/python3
# -*- coding:utf-8 -*-

# precise devision regardless of number type (res all in float)
from __future__ import division
import os
import gc
import re
import math
import timeit
# progress bar, wrap it around an iterator and it'll work
from tqdm import tqdm, trange
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import pandas as pd
import numexpr as ne

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
        print('\n[%0.8fs] %s()\n' % (elapsed, name))
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
        del self.ctts
        gc.collect()
        self.sims = self.simsCal()
        del self.wVecs
        gc.collect()
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

        print("Extracting contexts from corpora...")
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
        return ctts

    # to be written
    @clock
    def vecsWeight(self):
        # RelFreq weight function
        def RelFreq():
            wVecs = {}
            # convert the word vectors into pandas DataFrame format
            # and calculate the RelFreq of each context
            for cat in self.cats:
                vecs = pd.DataFrame(self.ctts[cat]).fillna(0)
                del self.ctts[cat]
                gc.collect()
                vecsKs = vecs.columns
                wordCounts = vecs.loc['WORD_COUNT']
                vecs.drop('WORD_COUNT', inplace=True)
                wVecs[cat] = (ne.evaluate('vecs/wordCounts'), vecsKs)
            return wVecs

        print("Weighting word vectors...")
        # this part trims off words and contexts that occur too rarely
        # it is said to be able to boost up the speed and increase accuracy
        for cttsCATk, cttsCATv in self.ctts.items():
            """
            res = {}
            for k, v in cttsCATv.items():
                # word count threshold
                if v['WORD_COUNT'] > len(self.corpsDir):
                    res[k] = {kk: vv for kk, vv in v.items() if kk !=
                              'WORD_COUNT' and vv > math.ceil(len(self.corpsDir)/10)}  # context count threshold
            self.ctts[cttsCATk] = res
            """
            # functions by creating a new main dict and copy everything relavant
            self.ctts[cttsCATk] = {
                k: v for k, v in cttsCATv.items() if v['WORD_COUNT'] > len(self.corpsDir)}
        return RelFreq()

    # Calculate words sims from weighted word vectors
    # output format:
    # {A  :
    #        (array
    #            ([[1.        , 0.63951928, 0.44219541, ..., 0.05904099, 0.20463049, 0.59703734],
    #              [0.63951928, 1.        , 0.55056397, ..., 0.02282177, 0.03061862, 0.56313762],
    #              [0.44219541, 0.55056397, 1.        , ..., 0.05255883, 0.        , 0.39399891],
    #              ...,
    #              [0.05904099, 0.02282177, 0.05255883, ..., 1.        , 0.        , 0.        ],
    #              [0.20463049, 0.03061862, 0.        , ..., 0.        , 1.        , 0.13968606],
    #              [0.59703734, 0.56313762, 0.39399891, ..., 0.        , 0.13968606, 1.        ]]
    #            ),
    #         Index(['obligatoire', 'ferme', 'laitier', 'adulte', 'biologique', 'plein',
    #                'premier', 'anglais', 'impatient', 'deux', ..., 'propriétaire', 'laxiste',
    #                'probatoire', 'ascendant', 'palpitant', 'manquer', 'thaonnais', 'mordant',
    #                'Elit', 'trail'], dtype='object', length=2797)
    #        ),
    # ADV  : ...,
    # ...,
    # }
    @clock
    def simsCal(self):
        def Cosine():
            sims = {}
            # go over CATs one by one
            for cat in self.cats:
                print("Processing CAT %s..." % cat)
                print("Converting into SciPy CSC sparse matrix... 1/4", end="\r")
                # convert into scipy sparse matrix for better cal performance
                wVecs = sp.csc_matrix(self.wVecs[cat][0])
                wVecsKs = self.wVecs[cat][1]
                del self.wVecs[cat]
                gc.collect()
                print("Converting into SciPy CSC sparse matrix... DONE")
                print("Calculating vectors' norms... 2/4", end="\r")
                # calculate the norm of all weigted word vectors and get their reciprocals
                # the result is a 1D array
                with np.errstate(divide="ignore"):
                    norms = np.divide(1, la.norm(wVecs, axis=0))
                # get the norms array's outer product, which is the numerator of our cosine function
                # the result is a 2D array (n x n where n the number of word vectors)
                normsOuter = np.outer(norms, norms)
                print("Calculating vectors' norms... DONE")
                print("Calculating vectors' dot product... 3/4", end="\r")
                # get the word vectors array(converted automatically from DataFrame by numpy)'s
                # outer product, which is the denominator(reciprocal) of our cosine function
                # the result is a 2D array (n x n where n the number of word vectors)
                # and restore into dense matrix for numexpr compatibility
                wVecsDot = wVecs.T.dot(wVecs).todense()
                del wVecs
                gc.collect()
                print("Calculating vectors' dot product... DONE")
                print("Calculating vectors' cosine similarities... 4/4", end="\r")
                # simsCat= numerator / denominator
                #        = numerator x denominator(reciprocal)
                # speed up with numexpr over numpy
                simsCat = ne.evaluate("wVecsDot*normsOuter")
                del wVecsDot, normsOuter
                gc.collect()
                print("Calculating vectors' cosine similarities... DONE")
                # append result: (sims in n x n array, column names)
                sims[cat] = (simsCat, wVecsKs)
            return sims

        print("Calculating words similarities...")
        return Cosine()

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
    dist = AzDist(corpsDir[:1], "A")
    for key, value in dist.sims.items():
        # print all similarities results to be examined
        print(key, ' : ', value, '\n\n')
        pass
