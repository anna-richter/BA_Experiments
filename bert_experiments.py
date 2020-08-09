import pandas as pd
import numpy as np
import pickle
import torch
from pytorch_pretrained_bert import BertTokenizer
from scipy.spatial.distance import cdist

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

class Bert_experimentor:
    def __init__(self, data: pd.DataFrame, model: object, model_name:str, layers):
        self.data = data
        self.model = model
        self.model_name = model_name
        self.layers = layers
        if self.model_name == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def get_bert_dict(self):
        bert_dict = {}
        lookup_tokens = []
        lookup_embeddings = []
        for i in self.data.columns:
            bert_dict[i] = {}
            self.data[i].dropna(inplace = True)
            for j in range(len(self.data[i])):
                bert_dict[i][j] = {}
                bert_dict[i][j]["text"] = self.data[i][j]
                tokens = mark_and_tokenize(str(self.data[i][j]), self.tokenizer)
                bert_dict[i][j]["tokens"] = tokens
                # we only want to add the words to the look up table, not the CLS and the SEP markers
                words_only = tokens[1:-1]
                for word in words_only:
                    lookup_tokens.append(word)
                ids = convert_to_ids(tokens, self.tokenizer)
                token_tensor, segment_tensors = convert_to_torch(ids, segments(ids))
                # get full embedding of Bert Model, embeddings has shape nr.of tokens x 13 x 768
                embeddings = get_hidden_states(token_tensor, segment_tensors, self.model)
                # if a certain layer was specified, get the embeddings of the layer
                # shape should be (nr of tokens x 768)
                if self.layers in ["1","2","3","4","5","6","7","8","9","10","11","12"]:
                    layer_embeddings = embeddings[:, int(self.layers)]
                    # add embeddings to lookup-embeddings dict so that we can find the respective token again
                    for tensor in layer_embeddings[1:-1]:
                        lookup_embeddings.append(tensor.numpy())

                # if mean is selected, it returns the average embedding over all layers,
                # shape should be (nr of tokens x 768)
                elif self.layers == "mean":
                    layer_embeddings = torch.mean(embeddings, dim=1)
                    for tensor in layer_embeddings[1:-1]:
                        lookup_embeddings.append(tensor.numpy())

                # if max_pool is selected, the maximum value of all layers is selected
                # shape should be (nr of tokens x 1 x 768)
                elif self.layers == "max":
                    layer_embeddings,_ = torch.max(embeddings, dim=1)
                    for tensor in layer_embeddings[1:-1]:
                        lookup_embeddings.append(tensor.numpy())

                # if sum is selected, the embeddings are summed over all layers
                elif self.layers == "sum":
                    layer_embeddings = torch.sum(embeddings, dim=1)
                    for tensor in layer_embeddings[1:-1]:
                        lookup_embeddings.append(tensor.numpy())


                bert_dict[i][j]["layer_embedding"] = layer_embeddings

        return bert_dict, lookup_tokens, lookup_embeddings

def mark_and_tokenize(comment, tokenizer):
    marked = "[CLS] " + comment + " [SEP]"
    ergebnis = tokenizer.tokenize(marked)
    return ergebnis

def convert_to_ids(comment, tokenizer):
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


def get_hidden_states(token_tensor, segment_tensors, model):
    # Predict hidden states features for each layer
    with torch.no_grad():
        output = model(token_tensor, segment_tensors)
        hidden_states = output[2]
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    return token_embeddings

def concatenate(embeddings):
    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in embeddings:

        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    return token_vecs_cat


# take average of all vectors and find word closest to it (mit cos similarity?)

def find_most_similar(bert_dictionary, look_up_tokens, look_up_embeddings):
    most_similar_dict = {}
    for topic in bert_dictionary.keys():
        avg_sen = []
        for i in bert_dictionary[topic].keys():
            # take only the layer 1 embeddings for the tokens, not for CLS and SEP
            embeddings = bert_dictionary[topic][i]["layer_embedding"][1:-1]
            # calculate average embedding
            average = torch.mean(embeddings, dim=0)
            #average = torch.mean(torch.stack(embeddings), dim=0)
            avg_sen.append(average)
            #torch.mean(torch.stack(embeddings), dim=0)


        # find nearest neighbour via cosine similarity, compare average embedding to all embeddings
        average = torch.mean(torch.stack(avg_sen), dim=0)
        distances = cdist([average.numpy()], look_up_embeddings, "cosine")[0]
        # order the indices according to distance and reverse this
        top_indices = np.flip(np.argsort(distances))
        # now go through the top indices, and add them to the top 5 if they denote a word which is not denoted
        # so far, do this until you have 5 different words
        top_5_words = []
        top_5_indices = []
        for index in top_indices:
            if look_up_tokens[index] not in top_5_words:
                top_5_words.append(look_up_tokens[index])
                top_5_indices.append(index)
            if (len(top_5_words) >= 5 or len(top_5_indices) >= 5):
                break

        most_similar_dict[topic] = []
        #print("5 top similar words for:", topic, "\n")
        for i, j in zip(top_5_words, top_5_indices):
            most_similar_dict[topic].append((i, distances[j]))
            #print("word: ", i, "\t", "similarity: ", distances[j])
        #print("\n")
    return most_similar_dict