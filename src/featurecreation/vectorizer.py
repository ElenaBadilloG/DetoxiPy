import pandas as pd
import numpy as np
import torch.utils.data as tud
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfData(tud.Dataset):

    def __init__(self, dataset, ngram_range, vocab_size):
        """
        Class to extract a TF-IDF sparse-matrix given a an iterable (list, 
        series, etc) of texts. Implements default tokenizers under the hood. 
        
        :param dataset: Iterable containing the text dataset to construct the
                        TF-IDF
        :type dataset: iterable (list, pandas series, numpy series)
        :param ngram_range: tuple with the upper and lower range of ngrams 
                            for which TF-IDF numbers should be constructed.
                            Ex: (1, 2) means TF-IDF will be constructed for 
                            unigrams and bigrams
        :type ngram_range: tuple
        :param vocab_size: Top n most common words to be included for the TF-IDF 
                           matrix
        :type vocab_size: int
        """
        self.vectorizer = TfidfVectorizer(ngram_range = ngram_range, 
                                          max_features = vocab_size)
        self.tf_idf = self._vectorize(dataset)

    def _vectorize(self, dataset):
        """
        Private method to vectorize the input data
        
        :param dataset: Iterable containing the text dataset to construct the
                        TF-IDF
        :type dataset: iterable (list, pandas series, numpy series)
        :return: Sparse RF-IDF matrix
        :rtype: sparse np matrix
        """

        tfidf_matrix = self.vectorizer.fit_transform(dataset)
        return tfidf_matrix

    def __getitem__(self, ix):

        return self.tf_idf[ix, :]