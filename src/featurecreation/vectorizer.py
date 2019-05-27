import pandas as pd
import numpy as np
import torch.utils.data as tud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from abc import ABC, abstractmethod

class BaseVectorizer(ABC):

    @abstractmethod
    def _vectorize(self, dataset):
        pass
    
class FreqVectorizer(tud.Dataset, BaseVectorizer):

    def __init__(self, dataset, ngram_range, vocab_size, vect_type, 
                 tokenizer = None):
        """
        Class to extract a Bag of Words sparse-matrix given a an iterable (list, 
        series, etc) of texts. Implements default tokenizers under the hood. 
        
        :param dataset: Iterable containing the text dataset to construct the
                        BoW
        :type dataset: iterable (list, pandas series, numpy series)
        :param ngram_range: tuple with the upper and lower range of ngrams 
                            for which BoW numbers should be constructed.
                            Ex: (1, 2) means TF-IDF will be constructed for 
                            unigrams and bigrams
        :type ngram_range: tuple
        :param vocab_size: Top n most common words to be included for the BOW 
                           matrix
        :type vocab_size: int
        :param tokenizer: Tokenizer to be used to convert a sentence into tokens
        :type tokenizer: Callable
        """
        if vect_type.lower() == "tf-idf":    
            self.vectorizer = TfidfVectorizer(ngram_range = ngram_range, 
                                              max_features = vocab_size,
                                              tokenizer = tokenizer)
        elif vect_type.lower() == "bow":
            self.vectorizer = CountVectorizer(ngram_range = ngram_range, 
                                              max_features = vocab_size,
                                              tokenizer = tokenizer)
        self.freq_vect = self._vectorize(dataset)

    def _vectorize(self, dataset):
        """
        Private method to vectorize the input data
        
        :param dataset: Iterable containing the text dataset to construct the
                        BoW
        :type dataset: iterable (list, pandas series, numpy series)
        :return: Sparse BoW matrix
        :rtype: sparse np matrix
        """
        freq_term_matrix = self.vectorizer.fit_transform(dataset)
        return freq_term_matrix

    def __getitem__(self, ix):

        return self.freq_vect[ix, :]
