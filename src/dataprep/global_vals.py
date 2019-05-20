"""
This module contains global flags and dictionaries for use throughout the 
entire project. 
"""

SPL_SEQ_DICT = {"emojis": [r":\)", r":-\)", r":\(", r":-\(", r":-/", r":/", 
                           r"-_-", r":\|", r":-\|"],
                "proper nouns": [r"republican", r"democrat", r"trump", 
                                 r"clinton", r"hillary"],
                "spl_punctuations": [r"!", r"\?", r"\*", r","],
                "punctuations": [r"!", r"\"", r"#", r"\$", r"%", r"&", r"'",
                                 r"\\", r"\(", r"\)", r"\*", r"\+", r",", r"-",
                                 r"\.", r"/", r":", r";", r"<", r"=", r">", 
                                 r"\?", r"@", r"\[", r"]", r"\^", r"_", r"`", 
                                 r"\{", r"\|", r"}", r"~"]}

# EVS NOTE: PUNCTUATION MARKS USED HERE ARE DERIVED FROM COMMON PUNTUATIONS
# AND "STAND INS" FOR LETTERS.

# THESE RGX DONT SATISFY ALL TEST CASES

# Pure words, which arent interspersed by special characters - "hell" vs "he*l"
RGX_PURE_WORD = r"[a-zA-Z]{2,}"
RGX_PURE_WORD_UPPER = r"[A-Z]{2,}"
RGX_PURE_WORD_LOWER = r"[a-z]{2,}"

# Words which might be interspersed by special characters - "hell" vs "he*l"
RGX_WORD = r"[a-zA-Z*]{2,}"
RGX_WORD_UPPER = r"[A-Z*]{2,}"
RGX_WORD_LOWER = r"[a-z*]{2,}"

# Regular expressions for characters
RGX_CHAR_UPPER = r"[A-Z]"
RGX_CHAR_LOWER = r"[a-z]"
