import pandas as pd
from gensim.models import KeyedVectors
from dataprep.text_cleaning import TextPrep

# READING THE DATASET
data_path = "train.csv"
comment_column = "comment_text"

raw_data = pd.read_csv(data_path, nrows = 150)
raw_data = raw_data[comment_column]

# CLEANING THE INPUT TEXT
text_prepper = TextPrep()
clean_data = raw_data.apply(text_prepper.clean, rmCaps = True, 
                            mapPunct = True, clSpecial = True, 
                            spCheck = False, rmStop = False, 
                            stem = False, mpContract = True)

# VECTORISING THE INPUT TEXT
clean_data = clean_data.apply(text_prepper.tokenize)

loaded_kv = KeyedVectors.load("featurecreation/embeddings/small_sample_vector_100K_ft.kv", mmap = "r")
vectoriser = lambda lst :  [loaded_kv[wrd] for wrd in lst]
clean_data = clean_data.apply(vectoriser)

