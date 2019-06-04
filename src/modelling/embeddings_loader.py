from gensim.models import KeyedVectors
import numpy as np

class EmbeddingsLoader:

    def __init__(self, embed_type, wrd_to_ix_dict, pretrained_embed_path):
        """
        Class to load pretrained GloVe or Word2Vec/FastText embeddings into a 
        numpy lookup matrix, where each row index corresponds to a word, as 
        defined by the wrd_to_ix_dict dictionary. 

        Words which are not present in the original pretrained embedding will
        be assigned a 0-vector, resulting in no impact on the model. Unkown words 
        are also stored in a list for diagnostics purposes.   
        
        :param embed_type: Type of the pre-trained embeddings to load. Can 
                           take either "glove" or "word2vec"
        :type embed_type: str
        :param wrd_to_ix_dict: Word to index dictionary trained on the input
                               data corpora
        :type wrd_to_ix_dict: Dictionary
        :param pretrained_embed_path: Path of the pre-trained embeddings
        :type pretrained_embed_path: str
        """
        embd_mat, unk_wrd = self._build_matrix(embed_type = embed_type, 
                                               wrd_to_ix_dict = wrd_to_ix_dict,
                                               pretrained_embed_path = pretrained_embed_path)
        self.embeddings_matrix = embd_mat
        self.unknown_words = unk_wrd
    
    def _get_coefs(self, word, *arr):
        """
        <TODO ELENA - docstrings>
        
        :param word: [description]
        :type word: [type]
        :return: [description]
        :rtype: [type]
        """
        return word, np.asarray(arr, dtype='float32')
    
    def _load_glove_embeddings(self, glove_path):
        """
        <TODO ELENA - docstrings and test>
        
        :param glove_path: [description]
        :type glove_path: [type]
        :return: [description]
        :rtype: [type]
        """
        with open(glove_path) as f:
            return dict(self._get_coefs(*line.strip().split(' ')) for line in f)

    def _load_w2v_embeddings(self, w2v_path):
        """
        Private function to load the pretrained word 2 vector keyed vector
        object containing the embeddings matrix.
        
        :param w2v_path: Path containing the pretrained word2vec keyed vectors
        :type w2v_path: str
        :return: 
        :rtype: Gensim KeyedVector object
        """

        return KeyedVectors.load(w2v_path)

    def _build_matrix(self, embed_type, wrd_to_ix_dict, pretrained_embed_path):
        """
        Function to build the embeddings lookup matrix for the the NN 
        embeddings layer. 
        Words in the training set which aren't present in the vocabulary used 
        to train the embeddings are given zero vectors.   
        
        :param embed_type: Type of the pre-trained embeddings to load. Can 
                           take either "glove" or "word2vec"
        :type embed_type: str
        :param wrd_to_ix_dict: Word to index dictionary trained on the input
                               data corpora
        :type wrd_to_ix_dict: Dictionary
        :param pretrained_embed_path: Path of the pre-trained embeddings
        :type pretrained_embed_path: str
        :return: Numpy matrix containing the embeddings lookup table, and list 
                 of unknown words
        :rtype: np matrix, list
        """

        if embed_type == "glove":
            embedding_index = self._load_glove_embeddings(pretrained_embed_path)
            embedding_matrix = np.zeros((len(wrd_to_ix_dict) + 1, 300))
        elif embed_type == "word2vec":
            embedding_index = self._load_w2v_embeddings(pretrained_embed_path)
            embedding_matrix = np.zeros((len(wrd_to_ix_dict) + 1, 
                                         embedding_index.vector_size))

        unknown_words = []
    
        for word, i in wrd_to_ix_dict.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                unknown_words.append(word)
        return embedding_matrix, unknown_words