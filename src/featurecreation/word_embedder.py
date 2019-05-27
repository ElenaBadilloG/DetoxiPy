import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
from nltk.util import ngrams
import pickle

class CbowEmbedder(nn.Module):

    def __init__(self, vocab_size, embeddings_dim, context_size, 
                 learning_rate = 0.001):
        """
        Class used to train an embedding layer given a corpus, or load
        a pretrained embedding layer, and use the embedding layer to extract
        word embeddings.
        
        :param vocab_size: Desired size of the vocabulary. Note that it 
                           should align with the vocabulary size of the 
                           VocabHelper
        :type vocab_size: int
        :param embeddings_dim: Required dimensionality of the embedding space
        :type embeddings_dim: int
        :param context_size: Size of the "context" window around the target 
                             word. Note that this is a symmetric window around 
                             the text, so a context of 2 means that CBOWs are
                             trained with 2 words to the either side of the 
                             target word 
        :type context_size: int
        :param learning_rate: Learning rate for the SGD optimiser, 
                              defaults to 0.001
        :type learning_rate: float, optional
        """
        super(CbowEmbedder, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embeddings_dim, 
                                        2*context_size)
        self.lin1 = nn.Linear(2*context_size*embeddings_dim, 128)
        self.lin2 = nn.Linear(128, vocab_size)

        self.loss_fn = nn.NLLLoss()
        self.optimiser = optim.SGD(self.parameters(), lr = learning_rate)
        self.context_size = context_size
        self.loss_history = []

    def make_context_tgt_pairs(self, text_series, text_prepper):
        """
        Function which given an iterable of texts, iterates through each
        one, builds the context-target ngrams across all the iterables and
        stores in a list of tuples. 
        
        :param text_series: Iterable containing the sentences of text 
                            constituting the corpora
        :type text_series: iterable of text
        :param text_prepper: Text data preparation object form 
                             dataprep/data_prep.py
        :type text_prepper: TextPrep object
        :return: List of tuples containin the context-target pairs
        :rtype: List of tuples
        """
        n = self.context_size
        data = []
        for sentence in text_series:

            sentence_tokens = text_prepper.tokenize(sentence)
            sentence_ngrams = list(ngrams(sentence_tokens, (2*n) + 1))
            for ngram in sentence_ngrams:
                ngram_lst = list(ngram)
                target = ngram_lst.pop(n)
                context = ngram_lst
                data.append((context, target))

        return data

    def _forward(self, input_data):
        """
        Private function to run the forward pass of the CBOW word embedding
        network. 
        
        :param input_data: Word index vector of the input data
        :type input_data: tensor of word indices
        :return: tensor of log probabilities
        :rtype: tensor
        """
        print(input_data)
        embeds = self.embed_layer(input_data).view((1, -1))
        out = F.relu(self.lin1(embeds))
        out = self.lin2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs
    
    def _train_epoch(self, input_ngrams, vocab_helper):
        """
        Private function to train a single epoch.
        
        :param input_ngrams: word-context ngrams to train the model
        :type input_ngrams: list of tuples containing word-context pairs
        :param vocab_helper: Vocabulary helper object trained on the same 
                             corpus as the input data
        :type vocab_helper: VocabHelper object
        """
        total_loss = 0
        for context, target in input_ngrams:

            # Setting up tensors and gradients
            self.zero_grad()
            context_ix = torch.tensor([vocab_helper.word_to_ix.get(w, 0) 
                                        for w in context], dtype = torch.long)
            target_ix = vocab_helper.word_to_ix.get(target, 0)

            # Forward pass
            log_probs = self._forward(context_ix)
            loss = self.loss_fn(log_probs, torch.tensor([target_ix], 
                                                        dtype = torch.long))

            # Back propogation
            loss.backward()
            self.optimiser.step()
            total_loss += loss.item()
        
        print(total_loss)
        self.loss_history.append(total_loss)

    def train_model(self, input_ngrams, num_of_epochs, vocab_helper):
        """
        Function to train the embedding network for the desired number of 
        epochs. 
        
        :param input_ngrams: word-context ngrams to train the model
        :type input_ngrams: list of tuples containing word-context pairs
        :param num_of_epochs: Number of epochs for which the model should
                              be trained
        :type num_of_epochs: int
        :param vocab_helper: Vocabulary helper object trained on the same 
                             corpus as the input data
        :type vocab_helper: VocabHelper object
        """
        for i in range(0, num_of_epochs):
            self._train_epoch(input_ngrams = input_ngrams, 
                             vocab_helper = vocab_helper)

    def export_embed_layer(self, export_path):
        """
        Function to export the trained embedding layer as a pickle 
        dump.
        
        :param export_path: Path where the embedding layer should be dumped
        :type export_path: str
        """
        with open(export_path, "wb") as file_handle:
            pickle.dump(self.embed_layer, file_handle)
    
    def load_embed_layer(self, embed_layer_path):
        """
        Function to load pickled embedding layer containing pre-trained 
        weights. 
        
        :param embed_layer_path: Path from which pre-trained layer needs to be
                                 loaded.
        :type embed_layer_path: str
        """
        with open(embed_layer_path, "rb") as file_handle:
            self.embed_layer = pickle.load(file_handle)
    
    def get_embeddings(self, word_ixs):
        """
        Function to extract the embeddings corresponding to input word indices.
        The word to index conversion should be done using a VocabHelper trained 
        on the same style of input data.
        
        :param word_ixs: Tensor of word indices to be converted
        :type word_ixs: tensor
        :return: Tensor of the embedding vectors corresponding to the words
        :rtype: tensor
        """
        return self.embed_layer(word_ixs)