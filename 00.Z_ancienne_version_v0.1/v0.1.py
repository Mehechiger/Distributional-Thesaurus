#!/usr/bin/python3
# -*- coding:utf-8 -*-


def contextsExtractor(corpusDirectory, columnsToExtract="2|3", catToExtract="A|ADV|N|V"):
    columnsToExtract = [int(columnStr)
                        for columnStr in columnsToExtract.split("|")]
    catToExtract = catToExtract.split("|")
    for cat in catToExtract:
        locals()["contexts"+cat] = {}

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


if __name__ == "__main__":
    print(contextsExtractor("estrepublicain.extrait-aa.19998.outmalt"))

"""




                    exec("contextStartValue=contexts{}.get(lineSplit[2],0)".format(lineSplit[3]))
                    exec("contexts{}[lineSplit[2]]=contextStartValue+1".format(lineSplit[3]))
                    """
