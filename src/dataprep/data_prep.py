import pandas as pd
import numpy as np 
import string
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class DataPrep:
    def __init__(self):
        self.tokenizer = TweetTokenizer()

    def tokenize(self, text):
        tknzr = self.tokenizer
        return tknzr.tokenize(text)

    def stemmer(self, text):
        text = self.tokenize(text)
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        return " ".join(stemmed_words)

    def rm_whitespace(self, text):        
        spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', \
            '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']
        for space in spaces:
            text = text.replace(space, ' ')
        text = text.strip()
        text = re.sub('\s+', ' ', text)
        return text

    def rm_punct(self, text):
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        for p in punct:
            text = text.replace(p, ' ')     
        return text

    def map_punct(self, text):
        # adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
        mapping = {"_":" ", "`":" "}
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        for p in mapping:
            text = text.replace(p, mapping[p])    
        for p in punct:
            text = text.replace(p, ' ')     
        return text

    def rm_stopword(self, text):
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops and len(w) >= 3]
        return " ".join(text)
    
    def lower_str(self, text):
        return text.lower()

    def clean(self, text, rmCaps, rmPunct, mapPunct, rmStop, stem):
        '''
        1. Remove whitespace
        2. Tokenize
        3. Option to lower
        4. Option to rm punctuation
        5. Option to lemmatize/stem
        '''
        
        if rmCaps == True:
            text = self.lower_str(text)
        if rmPunct == True:
            text = self.rm_punct(text)
        if mapPunct == True:
            text = self.map_punct(text)
        if rmStop == True:
            text = self.rm_stopword(text)
        if stem == True:
            text = self.stemmer(text)
        text = self.rm_whitespace(text)
        
        return text