import pandas as pd
import numpy as np
import pickle
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from nltk.tokenize.treebank import TreebankWordDetokenizer
from bert_experiments import Bert_experimentor

class Formatter:
    def __init__(self, model_name: str, data: pd.DataFrame):
        self.model_name = model_name
        self.data = data

    def format_data(self):
        if self.model_name in ["Fasttext", "Fasttext_gensim" "Word2Vec" ,"GloVe"]:
            return word_embedding_formatting(self.data)

        elif self.model_name in ["BERT", "ROBERTA", "ALBERT"]:
            return sentence_embedding_formatting(self.data)

        elif self.model_name == "test":
            return sentence_embedding_formatting(self.data, self.model)
        else:
            print("fehler duh")


def word_embedding_formatting(data):
    for topic in data.columns:
        data[topic].dropna(inplace=True)
        for comment in range(len(data[topic])):
            #data[topic][comment] = list(data[topic][comment])
            pass

    return data

def sentence_embedding_formatting(data, model):
    item = Bert_experimentor(data, model)
    return item.get_bert_dict()