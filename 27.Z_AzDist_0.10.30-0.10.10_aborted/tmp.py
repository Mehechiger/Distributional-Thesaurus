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
from bidict import bidict
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
        """
        #############################
        ######## DEBUG ONLY! ########
        self.cats = ["A", ]
        self.corpsDir = self.corpsDir[:1]
        """
        self.n = 5
        ######## DEBUG ONLY! ########
        #############################
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
        def ExtractNGrm(corpDir, n):
            with open(corpDir, 'r') as corpFile:
                try:
                    corpus = re.compile(
                        '(?:\n)|(?:(?:[^\t]*\t[^\t]*\t)(.*?)\t(.*?)\t.*\n)').findall(corpFile.read())
                except:
                    print("Error: corpus file %s unusable, has passed." % corpDir)
                    corpus = []

            for i in trange(len(corpus)):
                lemmaI, catI = corpus[i]
                if catI in self.cats and not any(char.isdigit() for char in lemmaI):
                    cKeys = ctts['cKeys%s' % catI]
                    rKeys = ctts['rKeys%s' % catI]
                    try:
                        cI = cKeys[lemmaI]
                        ctts[catI][0, cI] += 1
                    except KeyError:
                        cI = -cKeys['count']
                        cKeys['count'] -= 1
                        cKeys[lemmaI] = cI
                        ctts[catI][0, cI] += 1

                    passPrev, passNext = False, False
                    for j in range(1, n):
                        if passPrev == True:
                            pass
                        elif i-j == -1 or corpus[i-j] == ('', ''):
                            k = n-j+1
                            while k > 0:
                                prevGrm = '%s%s%s' % ('-', str(n-k+1), '^')
                                try:
                                    ctts[catI][rKeys[prevGrm], cI] += 1
                                except KeyError:
                                    rI = rKeys['count']
                                    rKeys['count'] += 1
                                    rKeys[prevGrm] = rI
                                    ctts[catI][rI, cI] += 1
                                k -= 1
                            passPrev = True
                        elif corpus[i-j][1] in ['A', 'ADV', 'N', 'V']:
                            # else:
                            if any(char.isdigit() for char in corpus[i-j][0]):
                                prevGrm = '(NBS)'
                            # elif corpus[i-j][1] == 'PONCT':
                            #    prevGrm = '(PONCT)'
                            else:
                                prevGrm = "".join(corpus[i-j])
                            prevGrm = '%s%s%s' % ('-', str(j), prevGrm)
                            try:
                                ctts[catI][rKeys[prevGrm], cI] += 1
                            except KeyError:
                                rI = rKeys['count']
                                rKeys['count'] += 1
                                rKeys[prevGrm] = rI
                                ctts[catI][rI, cI] += 1

                        if passNext == True:
                            pass
                        elif i+j == len(corpus) or corpus[i+j] == ('', ''):
                            k = n-j+1
                            while k > 0:
                                nextGrm = '%s%s%s' % ('+', str(n-k+1), '$')
                                try:
                                    ctts[catI][rKeys[nextGrm], cI] += 1
                                except KeyError:
                                    rI = rKeys['count']
                                    rKeys['count'] += 1
                                    rKeys[nextGrm] = rI
                                    ctts[catI][rI, cI] += 1
                                k -= 1
                            passNext = True
                        elif corpus[i+j][1] in ['A', 'ADV', 'N', 'V']:
                            # else:
                            if any(char.isdigit() for char in corpus[i+j][0]):
                                nextGrm = '(NBS)'
                            # elif corpus[i+j][1] == 'PONCT':
                            #    nextGrm = '(PONCT)'
                            else:
                                nextGrm = "".join(corpus[i+j])
                            nextGrm = '%s%s%s' % ('+', str(j), nextGrm)
                            try:
                                ctts[catI][rKeys[nextGrm], cI] += 1
                            except KeyError:
                                rI = rKeys['count']
                                rKeys['count'] += 1
                                rKeys[nextGrm] = rI
                                ctts[catI][rI, cI] += 1
                    cKeys = ctts['cKeys%s' % catI]
                    rKeys = ctts['rKeys%s' % catI]
            return

        print("Extracting contexts from corpora...")
        # initialize the main dict, creating dicts of each category needed
        ctts = {}
        for cat in self.cats:
            # ctts[cat] = sp.dok_matrix((300000, 300000))
            ctts[cat] = np.zeros((500000, 500000))
            #ctts[cat] = sp.lil_matrix((500000, 500000))
            ctts['cKeys%s' % cat] = bidict({'count': 0})
            ctts['rKeys%s' % cat] = {'count': 0}
        # go over the corpora file one by one
        # (only n-gram mode available for now. To be extended.)
        for i in trange(len(self.corpsDir)):
            # haven't decide yet how to pass in the n parameter of n-gram
            # so just put it here for now, by default bigram
            ExtractNGrm(self.corpsDir[i], self.n)
        for cat in self.cats:
            ctts[cat] = ctts[cat][:len(ctts['cKeys%s' % cat])-1, :len(
                ctts['cKeys%s' % cat])-1]
        print("\nExtracting contexts from corpora... DONE: %s corporus(a) processed." % len(
            self.corpsDir))
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
                vecs = sp.csc_matrix(self.ctts[cat])
                del self.ctts[cat]
                gc.collect()

                vecs = vecs.multiply(vecs[0, :] > 100)

                #mask = vecs[0, :] > len(self.corpsDir)
                mask = vecs[0, :] > 100
                #vecs = np.compress(mask, vecs, axis=1)
                vecs = vecs[mask]
                mask = vecs < 100
                vecs[mask] = 0

                self.ctts['cKeys%s' % cat].pop('count')
                vecsKs = dict(self.ctts['cKeys%s' % cat].inverse)
                del self.ctts['cKeys%s' % cat]
                gc.collect()

                wordCounts = vecs[0, :]
                vecs = vecs[1:, :]

                # wVecs[cat] = (ne.evaluate('vecs/wordCounts'), vecsKs)
                wVecsCat = ne.evaluate('vecs/wordCounts')
                """
                mask = wVecsCat < 0.0001
                wVecsCat[mask] = 0
                mask = wVecsCat > 0.6
                wVecsCat[mask] = 0
                """
                wVecs[cat] = (wVecsCat, vecsKs)
            print("Weighting word vectors... DONE")
            return wVecs

        print("Weighting word vectors...", end="\r")
        # this part trims off words and contexts that occur too rarely
        # it is said to be able to boost up the speed and increase accuracy
        '''
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
        '''
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
        print("Generating thesaurus for each word the first %s most similar word(s)..." %
              self.size, end='\r')
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
        output = ""
        for cat, items in dist.thes.items():
            output += "%s: \n\n" % cat
            for word, words in items.items():
                output += "%s: %s\n\n" % (word, words)
        f.write(output)
    print("Outputting thes file... DONE")
