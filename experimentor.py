import traceback
from gensim.models import KeyedVectors
from bert_experiments import Bert_experimentor
from bert_experiments import find_most_similar
from transformers import RobertaTokenizer, RobertaModel
# this class defines the experimental flow, which methods need to be called in which order etc
class Experimentor:
    def __init__(self, model, model_name, data, data_name, writer, layers):
        self.model = model
        self.model_name = model_name
        self.data = data
        self.data_name = data_name
        self.writer = writer
        self.layers = layers

    def run_experiment(self):
        if self.model_name in ["Fasttext", "Fasttext_gensim", "Word2Vec" ,"GloVe"]:
            titles = word_embeddings_experiment(self.data, self.data_name, self.model, self.writer)
            print(titles)
            self.writer.write_list(titles)
            self.writer.save()
            return titles

        elif self.model_name in ["BERT"]:
            experimentor = Bert_experimentor(self.data, self.model, self.model_name, self.layers)
            bert_dictionary, look_up_tokens, look_up_embeddings = experimentor.get_bert_dict()
            titles = find_most_similar(bert_dictionary, look_up_tokens, look_up_embeddings)
            self.writer.write_list(titles)
            self.writer.save()
            return titles

        else:
            print("fehler, kein Model angegeben")
            return None

def word_embeddings_experiment(data, data_name, model, writer):
    if data_name == "keywords":
        keyword_titles = {}
        for topic in data.columns:
            try:
                zwischenergebnis = model.most_similar_cosmul(positive= data[topic].values[0].split(" "),
                                                             negative=None, topn=5)
                keyword_titles[topic] = zwischenergebnis
            except KeyError as e:
                # Ignore the word if it does not exist.

                pass

        return keyword_titles


    elif data_name == "comments":
        comment_titles = {}
        for topic in data.columns:
            data[topic].dropna(inplace = True)
            comment_titles[topic] = []
            for comment in range(len(data[topic])):
                try:
                    zwischenergebnis = model.most_similar_cosmul(positive= data[topic][comment].split(" "),
                                                         negative=None, topn=5)
                    comment_titles[topic].append(zwischenergebnis)
                except KeyError as e:
                # Ignore the word if it does not exist.
                    pass

        # take list of 5 most similar words or each comment of the topic and add them to list
        titles = {}
        for topic in comment_titles.keys():
            wortliste = []
            for comment in comment_titles[topic]:
                for word in comment:
                    wortliste.append(word[0])
            titles[topic] = wortliste

        # calculate most similar words for list of most similar words for topics
        titles_final = {}
        for topic in titles.keys():
            zwischenergebnisse = []
            zwischenergebnisse.append(model.most_similar_cosmul(positive=titles[topic], negative=None, topn=5))
            titles_final[topic] = zwischenergebnisse


        return(titles_final)