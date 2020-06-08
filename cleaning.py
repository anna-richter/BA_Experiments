from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pickle
import nltk
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
#from porter2stemmer import Porter2Stemmer
import re
import os

nltk.download('stopwords')
stopwordList = stopwords.words('english')

nltk.download('wordnet')
nltk.download('punkt')

# check if the data was cleaned and saved under this configuration already, if not:
# clean data and save it
class Cleaner:
    def __init__(self, data_name:str, data: pd.DataFrame, cleaningsteps: list):
        self.data = data
        self.cleaningsteps = cleaningsteps
        self.path = f"./cleandata/{data_name}"
        for item in cleaningsteps:
            self.path += '_' + item.__name__
        self.path += ".csv"

    def clean_data(self):
        if os.path.exists(self.path):
            cleaned_data = pd.read_csv(self.path)

        else:
            if not os.path.exists(os.path.dirname(self.path)):
                os.makedirs(os.path.dirname(self.path))
            for operation in self.cleaningsteps:
                self.data = operation(self.data)
            cleaned_data = self.data
            cleaned_data.to_csv(self.path)

        return cleaned_data



def remove_html(txt):
    '''Remove HTML'''
    txt = BeautifulSoup(txt, 'lxml')
    return txt.get_text()

def remove_punctuation(surveyText):
    return "".join([i for i in surveyText if i not in string.punctuation or i == "-"])

def remove_stopwords(surveyText):
    return [w for w in surveyText if w not in stopwordList]

def restrict_vocab(surveyText):
    return [w for w in surveyText if w not in not_in_vocab]

def remove_anonymized(surveyText):
    return [w for w in surveyText if w != "anonymized"]

def basic(txt):
    #cleanedTxt = txt.apply(lambda x: remove_html(x))
    cleanedTxt = txt.apply(lambda x: remove_punctuation(x))
    cleanedTxt = cleanedTxt.apply(lambda x: word_tokenize(x.lower()))
    cleanedTxt = cleanedTxt.apply(lambda x: remove_anonymized(x))
    w_tokenizer = WhitespaceTokenizer()
    cleanedTxt = cleanedTxt.apply(lambda x: w_tokenizer.tokenize(x))
    return cleanedTxt

def stopwords(txt):
    cleanedTxt = cleanedTxt.apply(lambda x: remove_stopwords(x))
    return cleanedTxt

def vocab(txt):
    cleanedTxt = cleanedTxt.apply(lambda x: restrict_vocab(x))
    return cleanedTxt
