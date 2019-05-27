from collections import Counter
import pandas as pd
import numpy as np
import re
import pickle

def seq_counter(text, regex):
    """
    Utility function to count the number of occurrences of a regular 
    expression in a string
    
    :param text: String in which the regex counts need to be found
    :type text: str
    :param regex: string/regular expression to be counted
    :type regex: str
    :return: The number of occurrences of the regular expression in question
    :rtype: int
    """
    rgx = re.compile(regex)
    occurrences = re.findall(rgx, text)
    num_of_occurrences = len(occurrences)

    return num_of_occurrences #str(occurrences)

def set_seq_counter(text, set_of_seq):
    """
    Utility function to count the number of occurrences of each sequence
    in a set of sequences.
    
    :param text: String in which the counts are to be found
    :type text: str
    :param set_of_seq: Iterable (list, set or tuple) containing the
                       sequences who's counts are to be found
    :type set_of_seq: Iterable (list, set or tuple)
    :return: Dictionary containing the number of occurrences of each element
             in the input iterable
    :rtype: dict
    """
    seq_count_dict = {}
    for seq in set_of_seq:
        # print(seq + "\n")
        num_of_occurrences = seq_counter(text, seq)
        seq_count_dict[seq] = num_of_occurrences
    
    return sum(seq_count_dict.values())

class VocabularyHelper:

    def __init__(self, init_type, text_data_series = None, reqd_vocab_size = None, 
                 text_prepper = None, word_to_ix_path = None, vocab_path = None):
        """
        Class to provide a helper for keeping track of the vocabulary space 
        with a vocabulary set and a word-to-index dictionary mapping. 
        
        :param init_type: Flag denoting if vocabulary elements need to be 
                          trained on a new corpus ("train") or if they can
                          be loaded from pre-defined corpora ("load").

                          If "train", user needs to provide a series of text 
                          data from which vocabulary elements are to be extracted

                          If "load", user needs to provide paths to previously
                          defined pickles of word_to_ix dict and vocab set

        :type init_type: str
        :param text_data_series: Input data series from which the vocabulary 
                                 and mapping needs to be constructed, 
                                 defaults to None
        :type text_data_series: Iterable containing the strings constituting 
                                the corpus, optional
        :param reqd_vocab_size: The size of the vocabulary to be used. The top
                                n words are chosen to construct the vocabulary, 
                                defaults to None
        :type reqd_vocab_size: int, optional
        :param text_prepper: Text Preparation object form dataprep/textprep.py, 
                             defaults to None
        :type text_prepper: TextPrep object, optional
        :param word_to_ix_path: Path containing the pre-trained word to index
                                dictionary, defaults to None
        :type word_to_ix_path: str, optional
        :param vocab_path: Path containing the pre-trained vocabulary set,
                           defaults to None
        :type vocab_path: str, optional
        """

        if init_type.lower() == "train":
            self.vocab_size = reqd_vocab_size
            self.vocab = self._build_vocab_counter(text_data_series = text_data_series,
                                                text_prepper = text_prepper)
            
            self.word_to_ix = {k[0]: v+1 for v, k in enumerate(self.vocab)}
            self.word_to_ix["UNK"] = 0
            self.vocab = set(self.word_to_ix.keys())
        elif init_type.lower() == "load":
            self._load_vocab_elements(word_to_ix_path = word_to_ix_path,
                                      vocab_path = vocab_path)
        
    def _build_vocab_counter(self, text_data_series, text_prepper):
        """
        Private function to build a counter containing the words in the corpus
        and the counts of each word.
        
        :param text_data_series: Input data series from which the vocabulary 
                                 and mapping needs to be constructed 
        :type text_data_series: Iterable containing the strings constituting 
                                the corpus
        :param text_prepper: Text Preparation object form dataprep/textprep.py
        :type text_prepper: TextPrep object
        :return: Returns a counter containing the words and their frequencies 
                 in the total corpus
        :rtype: Counter
        """
        text_token_array = [text_prepper.tokenize(string) 
                                for string in text_data_series]
        text_token_array = [word for sublist in text_token_array 
                                    for word in sublist]
        vocab = Counter(text_token_array).most_common(self.vocab_size - 1)
        return vocab

    def export_vocab_element(self, element_type, export_path):
        """
        Function to export the word to index mapping as a pickle file.
        
        :param element_type: Type of the element to be exported - word_to_ix or vocab
        :type element_type: str
        :param export_path: Path where the word to index mapping should be stored
        :type export_path: str
        """

        if element_type.lower() == "word_to_ix":
            with open(export_path, "wb") as file_handle:
                pickle.dump(self.word_to_ix, file_handle)
        elif element_type.lower() == "vocab":
            with open(export_path, "wb") as file_handle:
                    pickle.dump(self.vocab, file_handle)

    def _load_vocab_elements(self, word_to_ix_path, vocab_path):
        """
        Private function to load pre-defined word to index dictionaries and 
        vocabulary sets.
        
        :param word_to_ix_path: Path containing the pre-trained word to index
                                dictionary
        :type word_to_ix_path: str
        :param vocab_path: Path containing the pre-trained vocabulary set
        :type vocab_path: str
        """
        with open(word_to_ix_path, "rb") as file_handle:
            self.word_to_ix = pickle.load(file_handle)
        
        with open(vocab_path, "rb") as file_handle:
            self.vocab = pickle.load(file_handle)
    