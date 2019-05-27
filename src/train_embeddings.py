import pandas as pd
from dataprep.data_prep import TextPrep 
from featurecreation.feat_create_utils import VocabularyHelper
from featurecreation.word_embedder import CbowEmbedder

# READING THE INPUT DATASET
text_column = "comment_text"

raw_data = pd.read_csv("train.csv", nrows = 1)
raw_data = raw_data[text_column]

# CLEANING THE INPUT TEXT
text_prepper = TextPrep()
clean_data = raw_data.apply(text_prepper.clean, rmCaps = False, 
                            mapPunct = True, clSpecial = True, 
                            spCheck = False, rmStop = True, 
                            stem = False, mpContract = True)

vocab_helper = VocabularyHelper(init_type = "train",
                                text_data_series = clean_data, 
                                reqd_vocab_size = 500, 
                                text_prepper = text_prepper)

# TRAINING THE EMBEDDING
cbw = CbowEmbedder(vocab_size = 500, embeddings_dim = 10, context_size = 2)
tst = cbw.make_context_tgt_pairs(text_series = clean_data, text_prepper = text_prepper)

cbw.train_model(input_ngrams = tst, num_of_epochs = 1, vocab_helper = vocab_helper)
cbw.export_embed_layer("embed.pkl")
