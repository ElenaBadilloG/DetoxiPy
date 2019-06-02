import pandas as pd
import numpy as np 
import string
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from dataprep.text_cleaning import *

class TextPrep:
    def __init__(self):
        self.tokenizer = TweetTokenizer()

    def tokenize(self, text):
        tknzr = self.tokenizer
        return tknzr.tokenize(text)

    def stemmer(self, text):
        text = self.tokenize(text)
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        return ' '.join(stemmed_words)

    # def remove(self, text, ws, punct):
    #     RM_LIST = []
    #     if ws == True:
    #         RM_LIST += SPACES
    #     if punct == True:
    #         RM_LIST += PUNCT
    #     for c in RM_LIST:
    #         text = text.replace(c, ' ')
    #     return text
    
    def clean_toks(self, text, rmStop, stem, mpContract):
        toks = self.tokenize(text)
        if rmStop == True:
            stops = set(stopwords.words("english"))
        if stem == True:
            stemmer = SnowballStemmer('english')
    
        cl_toks = []
        for t in toks:
            if rmStop == True:
                if len(t) > 3 and t not in stops:
                    if mpContract == True:
                        if t in CONTRACTION_MAP:
                            t = CONTRACTION_MAP[t]
                        if stem == True:
                            t = stemmer.stem(t)
                        cl_toks.append(t)
            else:
                if mpContract == True:
                    if t in CONTRACTION_MAP:
                        t = CONTRACTION_MAP[t]
                    if stem == True:
                        t = stemmer.stem(t)
                cl_toks.append(t)
        return ' '.join(cl_toks)
                    

    def rm_whitespace(self, text):        
        for space in SPACES:
            text = text.replace(space, ' ')
        text = text.strip()
        text = re.sub('\s+', ' ', text)
        return text

    def rm_punct(self, text):
        for p in PUNCT:
            text = text.replace(p, ' ')     
        return text

    def map_punct(self, text):
        for p in PUNCT_MAP:
            text = text.replace(p, PUNCT_MAP[p])    
        return self.rm_punct(text)

    def rm_stopword(self, text):
        stops = set(stopwords.words("english"))
        text = self.tokenize(text)
        text = [w for w in text if w not in stops and len(w) >= 3]
        return ' '.join(text)
    
    def lower_str(self, text):
        return text.lower()
    
    # def clean_contractions(self, text):
    #     for s in APOSTROPHES:
    #         text = text.replace(s, "'")
    #     text = self.tokenize(text)
    #     return ' '.join([CONTRACTION_MAP[w] for w in text if w in CONTRACTION_MAP else w])

    def clean_special_chars(self, text):
        for s in APOSTROPHES: 
            text = text.replace(s, "'")
        for s in SPECIAL_CHARS:
            text = text.replace(s, SPECIAL_CHARS[s])
        return text

    def correct_spelling(self, text):
        for word in SPELL_CORRECT:
            text = text.replace(word, SPELL_CORRECT[word])
        return text

    def clean(self, text, rmCaps, mapPunct, 
                    clSpecial, spCheck, rmStop, stem, mpContract):
        '''
        1. Remove whitespace
        2. Tokenize
        3. Option to lower
        4. Option to rm punctuation
        5. Option to lemmatize/stem
        '''
        if rmCaps == True:
            text = self.lower_str(text)
        if mapPunct == True:
            text = self.map_punct(text)
        if clSpecial == True:
            text = self.clean_special_chars(text)
        if spCheck == True:
            text = self.correct_spelling(text)

        text = self.clean_toks(text, rmStop, stem, mpContract)
        text = self.rm_whitespace(text)
        
        return text