#!/usr/bin/python3
# -*- coding:utf-8 -*-

import re


def corpusToContextsExtractor(corpusDir, columns="2|3", cats="A|ADV|N|V"):
    # lire le fichier
    def fileToBufferReader(corpusDir):
        with open(corpusDir, "r") as corpus:
            return corpus.readlines()

    # séparer le corpus en phrases
    def conllToSentencesSpliter(conll, columns):
        sentences = []
        sentence = []
        for line in conll:
            if line == "\n":
                sentences.append(sentence)
                sentence = []
            else:
                word = [line.split("\t")[column] for column in columns]
                sentence.append(word)
        return sentences

    def sentencesToContexts_CatParser(sentences, cats):
        def sentenceToContexts_CatParser(sentence, cats):
            prevLemma = ""
            contextsAllCats = {}
            for wordIndex in range(len(sentence)):
                word = sentence[wordIndex]
                if word[1] in cats:
                    contexts_CatVariableName = ""
                    exec("contexts_CatVariableName='contexts_%s'" %
                         re.sub(r'([\" \'])', r'\\\1', word[1]))
                    locals()[contexts_CatVariableName] = locals().get(
                        contexts_CatVariableName, {})
                    contextsAllCats[contexts_CatVariableName] = contextsAllCats.get(
                        contexts_CatVariableName, {})
                    lemmaVariableName = ""
                    exec("lemmaVariableName='%s'" %
                         re.sub(r'([\" \'])', r'\\\1', word[0]))
                    locals()[lemmaVariableName] = locals().get(
                        lemmaVariableName, {})
                    contextsAllCats[contexts_CatVariableName][lemmaVariableName] = contextsAllCats[contexts_CatVariableName].get(
                        lemmaVariableName, {})
                    if prevLemma != "":
                        contextsAllCats[contexts_CatVariableName][lemmaVariableName][(
                            -1, prevLemma)] = contextsAllCats[contexts_CatVariableName][lemmaVariableName].get((-1, prevLemma), 0)+1
                    if wordIndex+1 < len(sentence):
                        nextLemma = sentence[wordIndex+1][0]
                        contextsAllCats[contexts_CatVariableName][lemmaVariableName][(
                            +1, nextLemma)] = contextsAllCats[contexts_CatVariableName][lemmaVariableName].get((+1, nextLemma), 0)+1
                prevLemma = word[0]
            return contextsAllCats

        def contextsMerger(contexts1, contexts2):
            for key2, value2 in contexts2.items():
                if key2 in contexts1.keys():
                    contexts1[key2] += value2
                else:
                    contexts1[key2] = value2

        # créer des listes de vecteurs_mots-lexicaux-primaires (stockés en dict)
        # demandés par l'utilisateur, de type "contexts_A", "contexts_ADV", etc.
        # Le nom des variables est généré automatiquement avec locals()
        for cat in cats:
            locals()["contexts_"+cat] = {}

        # parser chaque phrase puis fusioner les contexts par CAT
        for sentence in sentences:
            contextsAllCats = sentenceToContexts_CatParser(sentence, cats)
            for cat in cats:
                contexts_CatVariableName = ""
                exec("contexts_CatVariableName=contexts_%s" % cat)
                if contexts_CatVariableName in contextsAllCats.keys():
                    contexts_CatVariableSelf = {}
                    exec("contexts_CatVariableName=contexts_%s" % cat)
                    contextsMerger(contexts_CatVariableSelf,
                                   contextsAllCats[contexts_CatVariableName])

        return contextsAllCats

    # obtenir la liste de colonnes à garder, en str
    columns = [int(column)
               for column in columns.split("|")]
    # obtenir la liste de CAT à garder, en str
    cats = cats.split("|")

    conll = fileToBufferReader(corpusDir)
    sentences = conllToSentencesSpliter(conll, columns)
    return sentencesToContexts_CatParser(sentences, cats)


"""
    with open(corpusDirectory, "r") as corpus:
        corpusContent = corpus.readlines()
        for lineIndex in range(len(corpusContent)):
            if corpusContent[lineIndex] == "\n" or lineIndex == 0:
                isBeg = True
            else:
                lineSplit = corpusContent[lineIndex].split("\t")
                if lineSplit[3] in catToExtract:
                    thisWord = lineSplit[2]
                    exec("if lineSplit[2] not in contexts%s : contexts%s[lineSplit[2]]={}" % (
                        lineSplit[3], lineSplit[3]))
                    if not isBeg:
                        exec("%s[(+1,%s)]=%s.get((+1,%s),0)+1" %
                             (prevVectorDirectory, thisWord, prevVectorDirectory, thisWord))
                        if lineIndex-1 >= 0:
                            exec("contexts%s[lineSplit[2]][(-1,%s)]=contexts%s[lineSplit[2]].get((-1,%s),0)+1" % (
                                lineSplit[3], prevWord, lineSplit[3], prevWord))
                    prevWord = lineSplit[2]
                    prevVectorDirectory = "contexts%s[lineSplit[2]]" % lineSplit[3]
                isBeg = False
    contexts = []
    for cat in catToExtract:
        exec("contexts.append(contexts%s)" % cat)
    return contexts


def similarityEvaluator():
    pass


def thesaurusGenerator():
    pass


"""

if __name__ == "__main__":
    print(corpusToContextsExtractor("estrepublicain.extrait-aa.19998.outmalt"))

"""

                    exec("contextStartValue=contexts{}.get(lineSplit[2],0)".format(
                        lineSplit[3]))
                    exec(
                        "contexts{}[lineSplit[2]]=contextStartValue+1".format(lineSplit[3]))
                    """
