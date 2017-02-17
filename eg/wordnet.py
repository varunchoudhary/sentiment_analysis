#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()
#print(lemmatizer.lemmatize("better",pos="a")) #good
#print(lemmatizer.lemmatize("best",pos="a"))   #best
#print(lemmatizer.lemmatize("gooder",pos="a")) #good
#print(lemmatizer.lemmatize("pythons",pos="a"))  #pythons

from nltk.corpus import wordnet
syns = wordnet.synsets("program")
# prints whole different meaning to any word
print(syns)
print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

#synonyms and antonyms

synonyms = []
antonyms = []

for syn in wordnet.synsets("love"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(synonyms)
print(antonyms)

# perentage of similarity between words

w1 = wordnet.synset("car.n.01")
w2 = wordnet.synset("plane.n.01")

p = w1.wup_similarity(w2)
print(p)