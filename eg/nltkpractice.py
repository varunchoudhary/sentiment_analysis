# tokenising

#from nltk.tokenize import word_tokenize

#text = "the punishment assigned to a defendant found guilty by a court, or fixed by law for a particular offence. Declare the punishment decided for (an offender)."
#print(word_tokenize(text))

#stopwords

#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

#text = "the punishment assigned to a defendant found guilty by a court, or fixed by law for a particular offence. Declare the punishment decided for (an offender)."

#stopwords = set(stopwords.words("english"))
#words = word_tokenize(text)
#filtered_sentence=[]

#for w in words:
  #  if w not in stopwords:
   #     filtered_sentence.append(w)
#print(filtered_sentence)

#stemming

#from nltk.stem.porter import PorterStemmer
#from nltk.tokenize import word_tokenize

#ps = PorterStemmer()
#exp = ["python","pythonic","pythonise"]

#for i in exp:
 #   print(ps.stem(i))

#part of speech tagging

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

traintext = state_union.raw("/home/varun/PycharmProjects/untitled/speechtrain.txt")
sampletext = state_union.raw("/home/varun/PycharmProjects/untitled/speechsample.txt")
costum_sent_tokenizer = PunktSentenceTokenizer(traintext)
tokenized = costum_sent_tokenizer.tokenize(sampletext)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
process_content()