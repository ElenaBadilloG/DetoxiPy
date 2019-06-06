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
        self.stemmer = SnowballStemmer('english')
        self.stopwords = set(stopwords.words("english"))

    def tokenize(self, text):
        tknzr = self.tokenizer
        return tknzr.tokenize(text)

    def clean_toks(self, text, rmStop, stem, mpContract):
        '''
        Function to handle text cleaning at the token level
        - Remove stop words
        - Map contractions
        - Stem using Snowball Stemmer
        '''
        toks = self.tokenize(text)
        cl_toks = []
        for t in toks:
            if rmStop == True:
                if len(t) > 3 and t not in self.stopwords:
                    if mpContract == True:
                        if t in CONTRACTION_MAP:
                            t = CONTRACTION_MAP[t]
                        if stem == True:
                            t = self.stemmer.stem(t)
                    cl_toks.append(t)
            else:
                if mpContract == True:
                    if t in CONTRACTION_MAP:
                        t = CONTRACTION_MAP[t]
                    if stem == True:
                        t = self.stemmer.stem(t)
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
        # for p in PUNCT_MAP:
        #     text = text.replace(p, PUNCT_MAP[p])
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)    
        # return self.rm_punct(text)
        return text
    
    def lower_str(self, text):
        return text.lower()
    
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
    
    def replace_idwords(self, text):
        exp = lambda kw: r'\b{}\b'.format(kw)
        for n in ID_WORDS:
            text = re.sub(exp(n), 'people', text, flags=re.IGNORECASE)
        for pn in PRONOUNS:
            text = re.sub(exp(pn), PRONOUNS[pn], text, flags=re.IGNORECASE)
        return text

    def clean(self, text, rm_caps, map_punct, cl_special, sp_check, replace_id,
              rm_stop, stem, mp_contract):
        '''
        1. Remove Caps
        2. Map and Remove Punctuation
        3. Clean Special Characters
        4. Correct Spelling Errors
        5. Replace Identity Words
        6. Clean Tokens: Remove Stopwords, Map Contractions, Stem
        7. Remove Whitespace
        '''
        if rm_caps == True:
            text = self.lower_str(text)
        if cl_special == True:
            text = self.clean_special_chars(text)
        if sp_check == True:
            text = self.correct_spelling(text)
        
        text = self.clean_toks(text, rm_stop, stem, mp_contract)        
        if map_punct == True:
            text = self.map_punct(text)
        if replace_id == True:
            text = self.replace_idwords(text)

        text = self.rm_whitespace(text)

        return text
    
def test_idwords():
    t0 = 'Sherry is a gay woman.'
    exp0 = 'Sherry is a people people.'
    t1 = 'I bought this pickle chip at Jewel Osco.'
    exp1 = 'I bought this pickle chip at Jewel Osco.'

    tp = TextPrep()    
    print(t0, tp.replace_idwords(t0))
    print(t1, tp.replace_idwords(t1))

def test(texts):
    tp = TextPrep()
    print('STOPWORDS: ', tp.stopwords)
    for t in texts:
        print(t)
        print(tp.clean(t, True, False, False, False, False, False, False))
        print(tp.clean(t, False, True, False, False, False, False, False))
        print(tp.clean(t, False, False, True, False, False, False, False))
        print(tp.clean(t, False, False, False, True, False, False, False))
        print(tp.clean(t, False, False, False, False, True, False, False))
        print(tp.clean(t, False, False, False, False, False, True, False))
        print(tp.clean(t, False, False, False, False, False, False, True))
        print(tp.clean(t, False, False, False, False, True, True, False))
        print(tp.clean(t, False, False, False, False, False, True, True))
        print(tp.clean(t, False, False, False, False, True, False, True))
        print(tp.clean(t, True, True, True, True, True, True, True))
        print()
        print('=========================================')
        print()
        
