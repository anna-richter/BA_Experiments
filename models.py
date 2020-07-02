import gensim.downloader as api
from gensim.models import KeyedVectors
from transformers import BertModel, BertTokenizer

def test():
    return "test_model"


def Fasttext():
    model = KeyedVectors.load_word2vec_format("C:/Users/Anna/Documents/GitHub/Bachelorarbeit/crawl-300d-2M-subword.vec")
    return model

def Fasttext_gensim():
    model = api.load("fasttext-wiki-news-subwords-300")
    return model

def GloVe():
    model = api.load("glove-twitter-200")
    return model

def Word2Vec():
    model = api.load("word2vec-google-news-300")
    return model

def BERT():
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    return model

def ROBERTA():
    pass

def ALBERT():
    pass