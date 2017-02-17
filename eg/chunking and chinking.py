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

    #chunking        chunkgram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?}"""
    #chunking        chunkgram = r"""Chunk: {<.*>+}
    #                                              }<VB.?|IN|DT|TO>+{"""
        nameEnt = nltk.ne_chunk(tagged,binary=true)
        print(nameEnt)
     #       chunkParser = nltk.RegexpParser(chunkgram)
      #      chunked = chunkParser.parse(tagged)
       #     print(chunked)


    except Exception as e:
        print(str(e))

process_content()