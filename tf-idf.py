import nltk
import numpy
from collections import OrderedDict
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#if-idf functions

def computetf(worddict,bow):
    tfdict = {}
    bowcount = len(bow)
    for word,count in worddict.items():
        tfdict[word] = count / float(bowcount)
    return tfdict

def computeidf(doclist):
    import math
    idfdict = {}
    N = len(doclist)

    idfdict = dict.fromkeys(doclist[0].keys(),0)

    #counting number of documents that contain a word w
    for doc in doclist:
        for word,val in doc.items():
            if val > 0:
                idfdict[word]+=1
    # divide N by denominator above, take the log of that
    for word, val in idfdict.items():
        idfdict[word] = math.log(N / float(val))

    return idfdict

def computetfidf(tfbow,idfs):
    tfidf={}
    for word,val in tfbow.items():
        tfidf[word] = val *idfs[word]
    return tfidf

def words(stringIterable):
    #upcast the argument to an iterator, if it's an iterator already, it stays the same
    lineStream = iter(stringIterable)
    for line in lineStream: #enumerate the lines
        for word in line.split(): #further break them down
            yield word


bagA = open("negative.txt","r").read()
bagB = open("positive.txt","r").read()
# words = []
# for p in bagA.split('\n'):
#     words.append(word_tokenize(p))
# print(words)
bowA = bagA.split("\n")
#removal of delimiter
bowAd = []
for p in bowA:
    net = re.sub(r'[?|$|#|.|,|.|!|:|%|&|\|[|]', r'', p)
    bowAd.append(re.sub(r'[^a-zA-Z0-9 ]', r'', net))

#convert lines to bag of words
bowAt = []
for word in words(bowAd):
        bowAt.append(word)

bowB = bagB.split("\n")
#removal of delimiter
bowBd = []
for p in bowB:
    # | ! | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    net= re.sub(r'[?|$|#|.|,|.|!|:|%|&|\|[|]', r'', p)
    bowBd.append(re.sub(r'[^a-zA-Z0-9 ]',r'',net))
#convert lines to bag words
bowBt = []
for word in words(bowBd):
        bowBt.append(word)
# delimiters = ['\n', '#', ',', '.', '?', '!', ':']


stopwords = set(stopwords.words("english"))

bowstopA = []
for p in bowAt:
     if p not in stopwords:
        bowstopA.append(p)


bowstopB = []
for p in bowBt:
     if p not in stopwords:
        bowstopB.append(p)


wordset = set(bowstopA).union(bowstopB)

worddictpos = dict.fromkeys(wordset,0)
worddictneg = dict.fromkeys(wordset,0)

for word in bowstopA:
    worddictneg[word]+=1

for word in bowstopB:
    worddictpos[word]+=1

tfbowA = computetf(worddictneg,bowstopA)
tfbowB = computetf(worddictpos,bowstopB)

idfs = computeidf([worddictneg,worddictpos])

tfidfbowA = computetfidf(tfbowA,idfs)
tfidfbowB = computetfidf(tfbowB,idfs)


# print((sorted(tfidfbowA, key=tfidfbowA.__getitem__), reverse=True))

negativearr = sorted(tfidfbowA, key=tfidfbowA.__getitem__, reverse=True)
positivearr = sorted(tfidfbowA, key=tfidfbowA.__getitem__, reverse=True)
# print(len(negativearr))
# print(len(positivearr))
document = []
for p in negativearr[:4000]:
    document.append((p,"neg"))
for p in positivearr[:6000]:
    document.append((p,"pos"))

numpy.random.shuffle(document)
# print(document)

all = nltk.FreqDist(document)
# print(all)
word_features = list(all.keys())

def find_features(document):
    words = word_tokenize(document)
    # print(words)
    features = {}
    for w in word_features:
        features[w] = (w in words)   
    return features

featuresets = [(find_features(rev), category) for (rev, category) in document]
print(len(featuresets))

trainingset = featuresets[:9900]
testingset = featuresets[9900:]
classifier = nltk.NaiveBayesClassifier.train(trainingset)
print("accuracy of nb classifier is",nltk.classify.accuracy(classifier,testingset)*100)


 