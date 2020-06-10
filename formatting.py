import pandas as pd
import numpy as np
import pickle
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Formatter:
    def __init__(self, model_name: str, data: pd.DataFrame):
        self.model_name = model_name
        self.data = data

    def format_data(self):
        if self.model_name == "Fasttext" or "Word2Vec" or "GloVe":
            return word_embedding_formatting(self.data)
        if self.model_name == "BERT":
            return sentence_embedding_formatting(self.data)


def word_embedding_formatting(data):
    for topic in data.columns:
        data[topic].dropna(inplace=True)
        for comment in range(len(data[topic])):
            data[topic][comment] = list(data[topic][comment])
    return data

def sentence_embedding_formatting(data):
    bert_dict = {}
    lookup_tokens = []
    lookup_embeddings = []
    for topic in data.columns:
        data[topic].dropna(inplace=True)
        bert_dict[topic] = {}
        for comment in range(len(data[topic])):
            bert_dict[topic][comment] = {}
            bert_dict[topic][comment]["text"] = data[topic][comment]
            one = mark_and_tokenize(str(data[topic][comment]))
            bert_dict[topic][comment]["tokens"] = one
            # we only want to add the words to the look up table, not the CLS and the SEP markers
            words_only = one[1:-1]
            for word in words_only:
                lookup_tokens.append(word)
            two = convert_to_ids(one)
            three = segments(two)
            token_tensor, segment_tensors = convert_to_torch(two, three)

        return (token_tensor,segment_tensors)

def mark_and_tokenize(comment):
    marked = "[CLS] " + comment + " [SEP]"
    ergebnis = tokenizer.tokenize(marked)
    return ergebnis

def convert_to_ids(comment):
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(comment)
    return indexed_tokens

def segments(comment):
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(comment)
    return segments_ids

def convert_to_torch(comment, segments):
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([comment])
    segments_tensors = torch.tensor([segments])
    return tokens_tensor, segments_tensors

