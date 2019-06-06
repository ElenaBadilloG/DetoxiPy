import pandas as pd
from dataprep.data_prep import TextPrep 
from featurecreation.feat_create_utils import VocabularyHelper
from gensim.models import FastText, Word2Vec
import multiprocessing
from time import time

# READING THE INPUT DATASET
text_column = "comment_text"

raw_data = pd.read_csv("train.csv")
raw_data = raw_data[text_column]

# CLEANING THE INPUT TEXT
text_prepper = TextPrep()
clean_data = raw_data.apply(text_prepper.clean, rmCaps = True, 
                            mapPunct = True, clSpecial = True, 
                            spCheck = False, replaceId = False, 
                            rmStop = False, stem = False, 
                            mpContract = True)

# TRAINING THE WORD2VEC
num_cores = multiprocessing.cpu_count()
embedder = Word2Vec(min_count = 20, window = 2, size = 64, sample = 6e-5, 
                    alpha = 0.03, min_alpha = 0.0007, negative = 20, 
                    workers = num_cores - 1)

# Tokenising sentences and building vocab
clean_data = clean_data.apply(text_prepper.tokenize)
t = time()
embedder.build_vocab(clean_data, progress_per = 10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# Training the model
t = time()
sample_data = clean_data.sample(frac = 0.75)
embedder.train(sample_data, total_examples = len(sample_data), epochs = 20,
               report_delay = 1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# Exporting the model weights

embedder.wv.save("featurecreation/embeddings/threequarters_sample_vector_noIDrepl_w2v.kv")
