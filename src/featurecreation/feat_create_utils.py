from collections import Counter
import pandas as pd
import numpy as np
import re

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

    def __init__(self, text_data_series, reqd_vocab_size, text_prepper):

        self.vocab_size = reqd_vocab_size
        self.vocab = self._build_vocab_counter(text_data_series = text_data_series,
                                               text_prepper = text_prepper)
        
        self.word_to_ix = {k[0]: v+1 for v, k in enumerate(self.vocab)}
        self.word_to_ix["UNK"] = 0
        self.vocab = set(self.word_to_ix.keys())
        

    def _build_vocab_counter(self, text_data_series, text_prepper):

        text_token_array = [text_prepper.tokenize(string) 
                                for string in text_data_series]
        text_token_array = [word for sublist in text_token_array 
                                    for word in sublist]
        vocab = Counter(text_token_array).most_common(self.vocab_size - 1)
        return vocab
