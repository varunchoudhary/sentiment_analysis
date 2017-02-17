
import nltk
import re 
import random
import pandas as pd
import math
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#if-idf functions##########################################################
def computetf(worddict, bow):
    tfdict = {}
    bowcount = len(bow)
    for word, count in worddict.items():
        tfdict[word] = count / float(bowcount)
    return tfdict


def computeidf(doclist):
    import math
    idfdict = {}
    N = len(doclist)

    idfdict = dict.fromkeys(doclist[0].keys(), 0)

    # counting number of documents that contain a word w
    for doc in doclist:
        for word, val in doc.items():
            if val > 0:
                idfdict[word] += 1
    # divide N by denominator above, take the log of that
    for word, val in idfdict.items():
        idfdict[word] = math.log(N / float(val))

    return idfdict


def computetfidf(tfbow, idfs):
    tfidf = {}
    for word, val in tfbow.items():
        tfidf[word] = val * idfs[word]
    return tfidf

###############################################################################

short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

# move this up here
all_words = []
all_wordspos = []
all_wordsneg = []
documents = []


#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

# for p in short_pos.split(" "):
    # words = p
###########################################################
for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
#removal of delimiter
    net= re.sub(r'[?|$|#|.|,|.|!|:|%|&|\|[|]', r'', p)
#convert lines to bag words
    words = word_tokenize((re.sub(r'[^a-zA-Z0-9 ]',r'',net)))
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_wordspos.append(w[0].lower())
            all_words.append(w[0].lower())


for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
#removal of delimiter     
    net= re.sub(r'[?|$|#|.|,|.|!|:|%|&|\|[|]', r'', p)

#convert lines to bag of words   
    words = word_tokenize(re.sub(r'[^a-zA-Z0-9 ]',r'',net))
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_wordsneg.append(w[0].lower())
            all_words.append(w[0].lower())

##########################################################
#converting it to set for tf-idf
print(len(all_wordspos))
wordset = set(all_wordspos).union(all_wordsneg)
print(len(wordset))
worddictpos = dict.fromkeys(wordset,0)
worddictneg = dict.fromkeys(wordset,0)
#if the word is in neg +1 in neg word and vica versa
for word in all_wordspos:
    worddictpos[word]+=1

for word in all_wordsneg:
    worddictneg[word]+=1

# calculate term weight frequency which is given by TF(word) = (Number of times word appears in a document) / (Total number of terms in the document).

tfpos= computetf(worddictpos,all_wordspos)
tfneg = computetf(worddictneg,all_wordsneg)
#calculate inverse document frequncy which is given by IDF(word) = log(Total number of documents / Number of documents with term word in it).
# print(tfpos,tfneg)
idfs = computeidf([worddictpos,worddictneg])
# print(idfs)
# total tf-idf score for the words in the negative and positive document
tfidfpos = computetfidf(tfpos,idfs)
tfidfneg = computetfidf(tfneg,idfs)


# print(pd.DataFrame(tfidfpos,tfidfneg))

negativearr = sorted(tfidfneg, key=tfidfneg.__getitem__, reverse=True)
positivearr = sorted(tfidfpos, key=tfidfpos.__getitem__, reverse=True)

print(len(all_words))
all = []
for p in negativearr:
	all.append(p)
for p in positivearr:
	all.append(p)
	
# print(all_words)

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


all = nltk.FreqDist(all_words)

# # print(all_words)
word_features = list(all.keys())[:5000]

# # print(len(word_features))
# # print(word_features)
save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    # print(words)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    # print(features)    
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]


save_feature_set = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(word_features, save_feature_set)
save_feature_set.close()

random.shuffle(featuresets)
print(len(featuresets))
testing_set = featuresets[6000:]
training_set = featuresets[:6000]

# #
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

###############
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

