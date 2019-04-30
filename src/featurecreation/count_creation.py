import pandas as pd
import re


data = pd.read_csv("/home/evsv/Documents/Spring 2019 Academics/Advanced ML/Final Project/DetoxiPy/src/featurecreation/train.csv")

comment_data = data["comment_text"]

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
    occurrences = re.findall(text, regex)
    num_of_occurrences = len(occurrences)

    return num_of_occurrences

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
        num_of_occurrences = seq_counter(text, seq)
        seq_count_dict[seq] = num_of_occurrences
    
    return seq_count_dict

# TESTING SCRIPTS BELOW

# import utils.global_vals as gv
# import re

# sample_txt = "WHAT THE F**K IS WRONG with  these  people?! Eh11 1219 a the's "
# sample_txt_2 = "*cant c*nt c*** ****"
# rgx = gv.RGX_PURE_WORD
# rgx_stp = "the"
# re.findall(rgx, sample_txt_2)



# print(gv.RGX_WORD_LOWER)