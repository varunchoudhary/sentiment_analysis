import nltk
import random
from nltk.tokenize import word_tokenize
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

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

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#for category in movie_reviews.categories():
#    for fileid in movie_reviews.fileids(category):
#        documents.append(list(movie_reviews.words(fileid)))

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words =nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["good"])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

fe = find_features(movie_reviews.words('neg/cv000_29416.txt'))

featuresets = [(find_features(rev), category) for (rev,category) in documents]

#or rev in documents:
  #  featuresets.append((find_features(rev)))

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("accuracy : ", (nltk.classify.accuracy(classifier, testing_set)))
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("mnb classifier accuracy status is: ",(nltk.classify.accuracy(MNB_classifier,training_set)))

#Gau_classifier = SklearnClassifier(GaussianNB())
#Gau_classifier.train(training_set)
#print("gaussian classifier acccuracy is:",(nltk.classify.accuracy(Gau_classifier,training_set)))

bernoulliNB_classifier = SklearnClassifier(BernoulliNB())
bernoulliNB_classifier.train(training_set)
print("bernoulli classifier acccuracy is:",(nltk.classify.accuracy(bernoulliNB_classifier,training_set)))

#LinearRegression, SGDClassifier

LinearRegression_classifier = SklearnClassifier(LinearRegression())
LinearRegression_classifier.train(training_set)
print("linear regression classifier acccuracy is:",(nltk.classify.accuracy(LinearRegression_classifier,training_set)))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDC classifier acccuracy is:",(nltk.classify.accuracy(SGDClassifier_classifier,training_set)))

#SVC, LinearSVC, NuSVC

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("svc classifier acccuracy is:",(nltk.classify.accuracy(SVC_classifier,training_set)))

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("linearSVC classifier acccuracy is:",(nltk.classify.accuracy(LinearSVC_classifier,training_set)))

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC classifier acccuracy is:",(nltk.classify.accuracy(NuSVC_classifier,training_set)))

voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  LinearRegression_classifier,
                                  bernoulliNB_classifier,
                                  MNB_classifier)

print("voted_classifiers accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print('classification',voted_classifier.classify(testing_set[0][0]),'confidence %',voted_classifier.confidence(testing_set[0][0])*100)