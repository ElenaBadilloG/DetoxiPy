import pandas as pd
from dataprep.data_prep import TextPrep 
from featurecreation.feat_create_utils import VocabularyHelper
from featurecreation.word_embedder import CbowEmbedder
from gensim.models import FastText
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
                            spCheck = False, rmStop = False, 
                            stem = False, mpContract = True)

# TRAINING THE WORD2VEC
num_cores = multiprocessing.cpu_count()
embedder = FastText(min_count = 20, window = 2, size = 64, sample = 6e-5, 
                    alpha = 0.03, min_alpha = 0.0007, negative = 20, 
                    workers = num_cores - 1)

# Tokenising sentences and building vocab
clean_data = clean_data.apply(text_prepper.tokenize)
embedder.build_vocab(clean_data, progress_per = 10000)
t = time()
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# Training the model
t = time()
sample_data = clean_data.sample(100000)
embedder.train(sample_data, total_examples = 100000, epochs = 30,
               report_delay = 1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# Exporting the model weights
embedder.wv.save("featurecreation/embeddings/small_sample_vector_100K_ft.kv")
