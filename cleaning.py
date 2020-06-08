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
            self.path += '_' + item

    def clean_data(self):
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        #for step in self.cleaningsteps:
         #   step(self.data)

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

def preprocessing(txt, punctuation=False, tokenize=False, stopwords=False, vocab=False,
                  anonymized=False, specialCharacter=False, numbers=False, singleChar=False):

    cleanedTxt = txt.apply(lambda x: remove_html(x))

    if punctuation:
        cleanedTxt = cleanedTxt.apply(lambda x: remove_punctuation(x))

    if tokenize:
        cleanedTxt = cleanedTxt.apply(lambda x: word_tokenize(x.lower()))

    if stopwords:
        cleanedTxt = cleanedTxt.apply(lambda x: remove_stopwords(x))

    if anonymized:
        cleanedTxt = cleanedTxt.apply(lambda x: remove_anonymized(x))

    # if vocab:
    #   cleanedTxt = cleanedTxt.apply(lambda x: restrict_vocab(x))

    if specialCharacter:
        '''Replacing Special Characters with space'''
        cleanedTxt = cleanedTxt.apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', str(x)))

    if numbers:
        '''Replacing Numbers with space'''
        cleanedTxt = cleanedTxt.apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))

    if singleChar:
        '''Removing words whom length is one'''
        cleanedTxt = cleanedTxt.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

    w_tokenizer = WhitespaceTokenizer()
    cleanedTxt = cleanedTxt.apply(lambda x: w_tokenizer.tokenize(x))

    if vocab:
        cleanedTxt = cleanedTxt.apply(lambda x: restrict_vocab(x))
    # for i in range(len(cleanedTxt)):
    #   cleanedTxt[i] = TreebankWordDetokenizer().detokenize(cleanedTxt[i])

    return cleanedTxt
