"""
This module contains global flags and dictionaries for use throughout the 
entire project. 
"""

SPL_SEQ_DICT = {"emojis": [":)", ":-)", ":(", ":-(", ":-/", ":/", "-_-", ":|", 
                            ":-|"],
                "proper nouns": ["republican", "democrat", "trump", "clinton", "hillary"],
                "spl_punctuations": ["!", "?", "*"]}

# EVS NOTE: PUNCTUATION MARKS USED HERE ARE DERIVED FROM COMMON PUNTUATIONS
# AND "STAND INS" FOR LETTERS.

# THESE RGX DONT SATISFY ALL TEST CASES

# Pure words, which arent interspersed by special characters - "hell" vs "he*l"
RGX_PURE_WORD = r"[a-zA-Z?!'][a-zA-Z?!]+"
RGX_PURE_WORD_UPPER = r"[A-Z?!'][A-Z?!']+"

# Words which might be interspersed by special characters - "hell" vs "he*l"
RGX_WORD = r"[a-zA-Z*?!][a-zA-Z]+[a-zA-Z*?!]+"
RGX_WORD_UPPER = r"[A-Z*?!']+"
RGX_WORD_LOWER = r"[a-z*?!']+"

