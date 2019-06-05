import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from dataprep.data_prep import TextPrep
from featurecreation.embeddings_loader import EmbeddingsLoader
from modelling.nn_models import LSTMModels

# FUNCTION DEFINITIONS
def build_datasets(train_path, data_sample_frac, text_colname, text_prepper,  
                   cln_rmCaps, cln_mapPunct, cln_clSpecial, cln_spChk, 
                   cln_replaceId, cln_rmStop, cln_stem, cln_mpContract, 
                   target_colname, aux_target_colnames, tokenizer, 
                   txt_token_seq_len, train_frac = 0.7):
    
    # READING IN THE DATASET
    df = pd.read_csv(train_path).sample(frac = data_sample_frac)
    
    # DIVIDING INTO TRAINING AND TESTING
    msk = np.random.rand(len(df)) < train_frac
    train = df[msk]
    test = df[~msk]

    # SPLITTING INTO X AND Y COLUMNS, APPLYING PREPROCESSING
    x_train = train[text_colname]
    x_train = x_train.apply(text_prepper.clean, rmCaps = cln_rmCaps, 
                            mapPunct = cln_mapPunct, clSpecial = cln_clSpecial, 
                            spCheck = cln_spChk, replaceId = cln_replaceId, 
                            rmStop = cln_rmStop, stem = cln_stem, 
                            mpContract = cln_mpContract)
    y_train = np.where(train[target_colname] >= 0.5, 1, 0)
    y_aux_train = train[aux_target_colnames]
    
    x_test = test[text_colname]
    x_test = x_test.apply(text_prepper.clean, rmCaps = cln_rmCaps, 
                          mapPunct = cln_mapPunct, clSpecial = cln_clSpecial, 
                          spCheck = cln_spChk, replaceId = cln_replaceId, 
                          rmStop = cln_rmStop, stem = cln_stem, 
                          mpContract = cln_mpContract)

    # PROCESSING TEXT SEQUENCES    
    tokenizer.fit_on_texts(list(x_train) + list(x_test))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train, maxlen = txt_token_seq_len)
    x_test = sequence.pad_sequences(x_test, maxlen = txt_token_seq_len)
    
    return x_train, x_test, y_train, y_aux_train, tokenizer, train, test

def build_model(x_train, y_train, x_test, y_aux_train, num_models, glove_matrix,
                model_type='LSTM'):
    
    x_train_torch = torch.tensor(x_train, dtype=torch.long)
    x_test_torch = torch.tensor(x_test, dtype=torch.long)
    y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]),
                                 dtype=torch.float32)
    train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
    test_dataset = data.TensorDataset(x_test_torch)

    all_test_preds = []

    for model_idx in range(NUM_MODELS):
        print('Model ', model_idx)
        seed_everything(1234 + model_idx)
        if model_type=='LSTM':
            model = LSTMModels(glove_matrix, y_aux_train.shape[-1]) 
        else:
            model = NeuralNetGRU(glove_matrix, y_aux_train.shape[-1])

        test_preds = train_model(model, train_dataset, test_dataset,
                                 output_dim=y_train_torch.shape[-1], 
                                 loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
        all_test_preds.append(test_preds)
        print()
    return all_test_preds

# DATAREAD PARAMETERS
train_path = "train.csv"
data_sample_frac = 0.1
text_colname = "comment_text"
target_colname = "target"
aux_target_colnames = ["target", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]
train_data_sample = 0.3
train_frac = 0.7

# DATAPROC PARAMETERS
text_prepper = TextPrep()
cln_rmCaps = True
cln_mapPunct = True
cln_clSpecial = True
cln_spChk = False 
cln_replaceId = False
cln_rmStop = False
cln_stem = False
cln_mpContract = True
tokenizer = text.Tokenizer()
txt_token_seq_len = 220

# EMBEDDINGS LOADER PARAMETERS
embed_type = "word2vec"
wrd_to_ix_dict = tokenizer.word_index
pretrained_embed_path = "featurecreation/embeddings/small_sample_vector_100K_ft.kv"

# BUILDING DATASET
data_result_set = build_datasets(train_path, data_sample_frac, text_colname, 
                                text_prepper, cln_rmCaps, cln_mapPunct, 
                                cln_clSpecial, cln_spChk, cln_replaceId, 
                                cln_rmStop, cln_stem, cln_mpContract, 
                                target_colname, aux_target_colnames, tokenizer,
                                txt_token_seq_len, train_frac)

x_train = data_result_set[0]
x_test = data_result_set[1]
y_train = data_result_set[2]
y_aux_train = data_result_set[3]
tokenizer_trained = data_result_set[4]
train = data_result_set[5]
test = data_result_set[6]

# LOADING EMBEDDINGS
embed_loader = EmbeddingsLoader(embed_type = embed_type, 
                                wrd_to_ix_dict = wrd_to_ix_dict,
                                pretrained_embed_path = pretrained_embed_path)

