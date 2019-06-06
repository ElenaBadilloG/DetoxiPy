import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import sys
sys.path.insert(0, '/Users/elenabg/')
import text_cleaner as pre

# Global params

GLOVE_EMBEDDING_PATH = '/Users/elenabg/Documents/6Q/AML/Project/glove.840B.300d.txt'
NUM_MODELS = 1
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220
MAX_FEATURES= None
FRAC = 0.015
TOKENIZER = text.Tokenizer()
TRAIN_PATH='/Users/elenabg/DetoxiPy/train.csv'
TEST_PATH='/Users/elenabg/DetoxiPy/test.csv'
IDENT_LIST = ['asian', 'atheist', 'bisexual', 'black', 'buddhist',  'christian', 'female', 'heterosexual',
'hindu', 'homosexual_gay_or_lesbian','intellectual_or_learning_disability','jewish','latino','male',
'muslim','other_disability','other_gender','other_race_or_ethnicity','other_religion', 'other_sexual_orientation',
              'physical_disability','psychiatric_or_mental_illness', 'transgender', 'white']


# Helper Funcs
def seed_everything(seed=1234): # for reproducibility
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words

def preprocess(data, rem_bias):
    '''
    Cleans comment text by:
    1) removing selected punctuation marks, 
    2) homogenezing contractions,
    3) homogenezing selected proper names,
    4) correcting selected misspellings
    '''

    data = data.astype(str).apply(lambda x: pre.clean_special_chars(x))
    data = data.astype(str).apply(lambda x: pre.clean_contractions_and_spelling(x))
    if rem_bias:
        data = data.astype(str).apply(lambda x: pre.replace_identities(x))
    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Model Definition: LSTM

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out



def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        scheduler.step()
        
        model.train()
        avg_loss = 0.
        
        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)            
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            
            
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        test_preds = np.zeros((len(test), output_dim))
    
        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
              epoch + 1, n_epochs, avg_loss, elapsed_time))

    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    
    else:
        test_preds = all_test_preds[-1]
    return test_preds


# Load data and build train/test sets

def build_datasets(tokenizer, rem_bias, frac=FRAC, train_frac=0.7, train_path=TRAIN_PATH,
                   test_path=TEST_PATH):
    
    df = pd.read_csv(train_path).sample(frac=frac)
    
    # divide into train and test
    msk = np.random.rand(len(df)) < train_frac
    train = df[msk]
    test = df[~msk]

    x_train = preprocess(train['comment_text'], rem_bias) # our own pre-processing pipeline goes here
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
    x_test = preprocess(test['comment_text'], rem_bias) # same
    tokenizer.fit_on_texts(list(x_train) + list(x_test))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    return x_train, x_test, y_train, y_aux_train, tokenizer, train, test


# Build Model  

def build_model(x_train, y_train, x_test, y_aux_train, NUM_MODELS, glove_matrix,
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
            model = NeuralNet(glove_matrix, y_aux_train.shape[-1]) 
        else:
            model = NeuralNetGRU(glove_matrix, y_aux_train.shape[-1])

        test_preds = train_model(model, train_dataset, test_dataset,
                                 output_dim=y_train_torch.shape[-1], 
                                 loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
        all_test_preds.append(test_preds)
        print()
    return all_test_preds


# Bias & Performance

def get_overall_perf(test, res, thresh=0.5):
    accuracy, precision, recall = None, None, None
    #test = test.loc[:, 'id':'comment_text']
    test['probs'] = res['prediction']
    test['preds'] = test['probs'].apply(lambda x: 1 if x >= thresh else 0)
    test['true'] = test['target'].apply(lambda x: 1 if x >= thresh else 0)
    test['correct'] = test['true']==test['preds']
    test['correct'] = test['correct'].apply(lambda x: 1 if x == True else 0)
    test1prec = test[test['preds'] == 1]
    accuracy = test['correct'].sum()/len(test) 
    lenp = len(test1prec)
    if lenp > 0:
        precision = test1prec['correct'].sum()/lenp
    print("Accuracy: {} \n Precision: {}".format(accuracy,precision))
    return test, accuracy, precision

def get_bias(test, precision, ident_collist=IDENT_LIST, thresh=0.5, wb=0.3, wp=0.7):
    
    def wav(bias, prec, wb, wp):
        return wb*(1-bias) + (wp*prec)
    
    test['identity']=(test[ident_collist]>=0.5).max(axis=1).astype(bool)

    test['identity'] = test['identity'].apply(lambda x: 1 if x else 0)
        
    testID = test[test['identity'] == 1]
    testNONID = test[test['identity'] == 0]
    
    testIDprec = testID[testID['preds'] == 1]
    accuracyID = testID['correct'].sum()/len(testID) 
    lenpid = len(testIDprec)
    if lenpid > 0:
        precID = testIDprec ['correct'].sum()/lenpid 
    
    testNONIDprec = testNONID[testNONID['preds'] == 1]
    accuracyNONID = testNONID['correct'].sum()/len(testNONID) 
    lenpnonid = len(testNONIDprec)
    if lenpnonid  > 0:
        precNONID = testNONIDprec['correct'].sum()/lenpnonid

    bias = precNONID - precID
    perf = wav(bias, precision,  wb, wp)
    print("Overall Precision: {} \n Bias: {}, \n Overall Performance: {}".format(precision, bias, perf))
    return test, perf, bias

# Main

def main (rem_bias=True, tokenizer = TOKENIZER, frac=FRAC, train_path=TRAIN_PATH, test_path=TEST_PATH):

    seed_everything()
    x_train, x_test, y_train, y_aux_train, tokenizer_trained, train, test = build_datasets(tokenizer, rem_bias, frac, train_path, test_path)

    max_features = MAX_FEATURES or len(tokenizer_trained.word_index) + 1 
    glove_matrix, unknown_words_glove = build_matrix(tokenizer_trained.word_index, GLOVE_EMBEDDING_PATH)

    print('\n unknown words (glove): ', len(unknown_words_glove))

    all_test_preds = build_model(x_train, y_train, x_test, y_aux_train, NUM_MODELS, glove_matrix)

    res = pd.DataFrame.from_dict({ 'id': test['id'], 'prediction': np.mean(all_test_preds,
                             axis=0)[:, 0]})
    test_expand, accuracy, precision = get_overall_perf(test, res)
    
return get_bias(test_expand, precision)