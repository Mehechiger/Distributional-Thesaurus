#!/usr/bin/python3
# -*- coding:utf-8 -*-

# precise devision regardless of number type (res all in float)
from __future__ import division
import os
import gc
import re
import sys
import argparse
import math
import timeit
from itertools import combinations
# progress bar, wrap it around an iterator and it'll work
from tqdm import tqdm, trange
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
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
    def __init__(self, corpsDir, cats, mode, wFunc, sFunc, size, path):
        # assignment of attributs
        self.cats, self.wFunc, self.sFunc, self.size = cats, wFunc, sFunc, int(
            size)
        self.n = int(mode[0]) if mode[-1] == "gram" else 0
        corpsDir = corpsDir+"/" if corpsDir[-1] != "/" else corpsDir
        # make sure we only take file with filename extenstion ".outmalt"
        self.corpsDir = ['%s%s' % (corpsDir, corpDir) for corpDir in os.listdir(
            corpsDir) if corpDir[-8:] == ".outmalt"]

        """
        #############################
        ######## DEBUG ONLY! ########
        self.cats = ["A", ]
        self.corpsDir = self.corpsDir[:1]
        ######## DEBUG ONLY! ########
        #############################
        """
        """
        """
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
        self.thes = self.thesGen()
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
                if catI in self.cats and not any(char.isdigit() for char in lemmaI):
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
                        elif corpus[i-j][1] in ['A', 'ADV', 'N', 'V']:
                            tmp = "".join(corpus[i-j])
                            if any(char.isdigit() for char in corpus[i-j][0]):
                                prevGrm = '(NBS)'
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
                        elif corpus[i+j][1] in ['A', 'ADV', 'N', 'V']:
                            tmp = "".join(corpus[i+j])
                            if any(char.isdigit() for char in corpus[i+j][0]):
                                nextGrm = '(NBS)'
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
            ctts = ExtractNGrm(self.corpsDir[i], ctts, self.n)

        print(
            "\nFiltering insufficient data and converting into 2D NumPy arry...", end="\r")
        # this part trims off words and contexts that occur too rarely
        # it is said to be able to boost up the speed and increase accuracy
        for cttsCATk, cttsCATv in ctts.items():
            filtered = {}
            for k, v in cttsCATv.items():
                # word count threshold
                if v['WORD_COUNT'] > 100:
                    filtered[k] = {kk: vv for kk, vv in v.items() if kk !=
                                   'WORD_COUNT' and vv > 100}  # context count threshold
                    if filtered[k] == {}:
                        filtered.pop(k)
            ctts[cttsCATk] = filtered

        # this part convert the dict(dict) format into a 2D np array
        for cat in self.cats:
            # create a 2D array filled with 0
            # Nb of rows varies according to the number of grams
            # Nb of columns corresponds to Nb of primary words
            cttsCAT = np.zeros((5000*(self.n-1), len(ctts[cat])))
            # stores primary words and their index
            cKeysCAT = list(ctts[cat].keys())
            # stores contexts and their index
            rKeysCAT = {'count': 0}
            # tells whether the Nb of different contexts exceeds the capacity of the matrix
            rStop = False

            for i in range(len(cKeysCAT)):
                # ctt, NbCtt correspond to one of all contexts and their Nb of occurance, of one word
                for ctt, NbCtt in ctts[cat][cKeysCAT[i]].items():
                    # if the context is already indexed, just fill the data into the matrix
                    try:
                        cttsCAT[rKeysCAT[ctt], i] = NbCtt
                    # if not:
                    except KeyError:
                        # if we already know that the matrix has reached its capacity, just ignore
                        if rStop:
                            continue
                        # if not yet, index new context
                        rKeysCAT[ctt] = rKeysCAT['count']
                        rKeysCAT['count'] += 1
                        # then try again filling in the data
                        try:
                            cttsCAT[rKeysCAT[ctt], i] = NbCtt
                        # if at this particular time the matrix reaches its capacity
                        # set rStop to True and delete the context that has just been indexed
                        except IndexError:
                            rStop = True
                            rKeysCAT.pop(ctt)
            del ctts[cat]
            gc.collect()

            # keep only the filled part of the matrix
            ctts[cat] = (cttsCAT[:len(rKeysCAT)-1, :], cKeysCAT)

        print("Filtering insufficient data and converting into 2D NumPy array... DONE")

        print("\nExtracting contexts from corpora... DONE: %s corporus(a) processed." % len(
            self.corpsDir))
        return ctts

    # to be written
    @clock
    def vecsWeight(self):
        # RelFreq weight function
        def RelFreq():
            wVecs = {}
            for cat in self.cats:
                wVecsCAT, cKeysCAT = self.ctts[cat]
                wordCounts = ne.evaluate('sum(wVecsCAT, axis=0)')

                # these lines filter off the abnormal high frequency contexts
                # the exact value to be determined with experiments
                mask = ne.evaluate('wVecsCAT/wordCounts') >= 0.9
                wVecsCAT[mask] = 0

                wordCounts = ne.evaluate('sum(wVecsCAT, axis=0)')
                wVecs[cat] = (ne.evaluate('wVecsCAT/wordCounts'), cKeysCAT)
            print("Weighting word vectors... DONE")
            return wVecs

        print("Weighting word vectors...", end="\r")
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
    #  ADV  : ...,
    #  ...,
    # }
    @clock
    def simsCal(self):
        def Cosine():
            sims = {}
            # go over CATs one by one
            for cat in self.cats:
                print("\tProcessing CAT %s..." % cat)
                print("\t\tConverting into SciPy CSC sparse matrix... 1/4", end="\r")
                # convert into scipy sparse matrix for better cal performance
                wVecs = sp.csc_matrix(self.wVecs[cat][0])
                wVecsKs = self.wVecs[cat][1]
                del self.wVecs[cat]
                gc.collect()
                print("\t\tConverting into SciPy CSC sparse matrix... DONE")
                print("\t\tCalculating vectors' norms... 2/4", end="\r")
                # calculate the norm of all weigted word vectors and get their reciprocals
                # the result is a 1D array
                with np.errstate(divide="ignore"):
                    norms = np.divide(1, la.norm(wVecs, axis=0))
                # get the norms array's outer product, which is the numerator of our cosine function
                # the result is a 2D array (n x n where n the number of word vectors)
                normsOuter = np.outer(norms, norms)
                print("\t\tCalculating vectors' norms... DONE")
                print("\t\tCalculating vectors' dot product... 3/4", end="\r")
                # get the word vectors array(converted automatically from DataFrame by numpy)'s
                # outer product, which is the denominator(reciprocal) of our cosine function
                # the result is a 2D array (n x n where n the number of word vectors)
                # and restore into dense matrix for numexpr compatibility
                wVecsDot = wVecs.T.dot(wVecs).todense()
                del wVecs
                gc.collect()
                print("\t\tCalculating vectors' dot product... DONE")
                print("\t\tCalculating vectors' cosine similarities... 4/4", end="\r")
                # simsCat= numerator / denominator
                #        = numerator x denominator(reciprocal)
                # speed up with numexpr over numpy
                simsCat = ne.evaluate("wVecsDot*normsOuter")
                del wVecsDot, normsOuter
                gc.collect()
                print("\t\tCalculating vectors' cosine similarities... DONE")
                # append result: (sims in n x n array, column names)
                sims[cat] = (simsCat, wVecsKs)
                print("\tProcessing CAT %s... DONE" % cat)
            print("Calculating words similarities... DONE")
            return sims

        print("Calculating words similarities...")
        return Cosine()

    @clock
    # generate the thesaurus with the first N(self.size) most similar words of each word
    # output format:
    # {"category1":
    #               {"word1":
    #                           {"wordRank1":simScore,
    #                           "wordRank2":simScore,
    #                           "wordRank3":simScore,
    #                           ...
    #                           },
    #               "word2":
    #                           {"wordRank1":simScore,
    #                           "wordRank2":simScore,
    #                           "wordRank3":simScore,
    #                           ...
    #                           },
    #               ...
    #               },
    # category2:
    #               {...
    #               },
    # ...
    # }
    def thesGen(self):
        print("Generating thesaurus for each word the first %s most similar word(s)...", end='\r')
        thes = {}
        for cat in self.cats:
            sims, ks = self.sims[cat]
            del self.sims[cat]
            gc.collect()
            # get the indexes of the first N(self.size) most similar words in the matrix
            indsTopN = np.argsort(ne.evaluate("-sims"))[:, 1:self.size+1]
            thesCAT = {}
            # use the indexes and the matrix keys(which are the words) to reconstruct
            # a dict based thesaurus
            for i in range(indsTopN.shape[0]):
                thesWord = {}
                for j in range(self.size):
                    thesWord[ks[indsTopN[i, j]]] = sims[i, indsTopN[i, j]]
                thesCAT[ks[i]] = thesWord
            thes[cat] = thesCAT
        print("Generating thesaurus for each word the first %s most similar word(s)... DONE" % self.size)
        return thes


# calling and testing
if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    # default to be modified when release
    argParser.add_argument(
        "-d", "--corpsdir",
        default="/Users/mehec/nlp/prjTAL_L3/donnees/",
        help="corpora directory")

    # the choices part generate all combinations of the 4 cats from 1 to 4 elms
    # the type part convert input eg. A,ADV to ["A", "ADV"]
    argParser.add_argument(
        "-c", "--cats",
        default=["A", "ADV", "N", "V"],
        choices=[set(ch) for comb in map(lambda y: combinations(
            ["A", "ADV", "N", "V"], y), range(1, 5)) for ch in set(comb)],
        type=lambda s: set(s.split(',')),
        help="categories to be analized(use comma, to seperate cats \
                                        and do not insert space between them, \
                                        eg. python3 AzDist.py - c A, V); \
                                        default is A,ADV,N,V")

    # the type part convert input eg. 3-gram to ["2", "gram"] or dep to ["dep", ]
    argParser.add_argument(
        "-m", "--mode",
        default=["2", "gram"],
        choices=[[str(n), "gram"] for n in range(2, 10)]+["dep", ],
        type=lambda s: s.split('-'),
        help='contexts extract mode(specify the n in case of ngram, \
                                        use dash - to seperate n and gram \
                                        and do not insert space between them, \
                                        eg. "python3 AzDist -m 5-gram"); \
                                        default is 2-gram')

    argParser.add_argument(
        "-wf", "--wfunc",
        default="RelFreq",
        choices=["RelFreq", ],
        help="weight function; default is RelFreq")

    argParser.add_argument(
        "-sf", "--sfunc",
        default="COS",
        choices=["COS", ],
        help="similariy function; default is COS")

    argParser.add_argument(
        "-s", "--size",
        default=10,
        type=int,
        help="number of similar words under one item; default is 10")

    # to be finished
    # more features to be added: file format as json, csv, etc.
    # the type part add automatically / if needed
    argParser.add_argument(
        "-p", "--path",
        default="",
        type=lambda s: "%s/" % s if len(s) != 0 and s[-1] != "/" else s,
        help="set output (abs) path; default is path of AzDist.py")

    args = argParser.parse_args()

    dist = AzDist(args.corpsdir, args.cats, args.mode,
                  args.wfunc, args.sfunc, args.size, args.path)

    print("Outputting thes file...", end="\r")
    # overwrite alert to be added
    with open("%sthesAzDist.txt" % args.path, "w") as f:
        f.write(str(dist.thes))
    print("Outputting thes file... DONE")
