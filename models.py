import gensim.downloader as api

def test():
    return "test_model"

def Fasttext():
    model = api.load("fasttext-wiki-news-subwords-300")
    return model

def GloVe():
    model = api.load("glove-twitter-200")
    return model

def Word2Vec():
    model = api.load("word2vec-google-news-300")
    return model

def BERT():
    pass

def ROBERTA():
    pass

def ALBERT():
    pass