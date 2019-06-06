from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np
from time import time

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NNModels(nn.Module, ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self):
        pass

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def train_model(self, train, test, loss_fn, output_dim, lr = 0.001, 
                    batch_size = 512, n_epochs = 4, 
                    enable_checkpoint_ensemble = True):
        
        # Initialising training components

        param_lrs = [{"param": param, "lr": lr} for param in self.parameters()]
        optimiser = torch.optim.Adam(param_lrs, lr = lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: 0.6**epoch)

        train_loader = torch.utils.data.DataLoader(train, 
                                                   batch_size = batch_size, 
                                                   shuffle = True)
        test_loader = torch.utils.data.DataLoader(test, 
                                                  batch_size = batch_size, 
                                                  shuffle = False)
        
        all_test_preds = []
        checkpoint_weights = [2**epoch for epoch in range(n_epochs)]

        # Iterating across epochs
        for epoch in range(n_epochs):
            
            start_time = time()
            scheduler.step()
            
            # In training mode
            self.train()
            avg_loss = 0
            for data in train_loader:
                
                x_batch = data[:-1]
                y_batch = data[-1]

                y_pred = self.forward(*x_batch)
                loss = loss_fn(y_pred, y_batch)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                avg_loss += loss.item()/len(train_loader)

            # In evaluation mode
            self.eval()
            test_preds = np.zeros(len(test), output_dim)

            for i, x_batch in enumerate(test_loader):
                y_pred = self.sigmoid(self.forward(*x_batch).detach().cpu().numpy())
                test_preds[i*batch_size:(i+1)*batch_size, :] = y_pred
            
            all_test_preds.append(test_preds)
            elapsed_time = time() - start_time
            print("Epoch {}/{} \t loss={:.4f} \t time={:.2f}s".format(
                  epoch + 1, n_epochs, avg_loss, elapsed_time))

        if enable_checkpoint_ensemble:
            test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    
        else:
            test_preds = all_test_preds[-1]
        
        return test_preds

class LSTMModels(NNModels):

    def __init__(self, embedding_matrix, max_features, num_aux_targets,  
                dense_hidden_units = None, lstm_units = 128):
        
        super().__init__()

        embed_size = embedding_matrix.shape[1] 
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        if dense_hidden_units is None:
            dense_hidden_units = 4*lstm_units

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)
        
        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)
    
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