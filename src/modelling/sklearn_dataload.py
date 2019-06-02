import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class DataPrep:
    def __init__(self, text_col, label_col):
        """
        Class to extract and prepare the data for modeling.
        Functionalities within this class are:
            1. Extract user specified columns from the dataframe
            2. Select a subsample of the feature dataset
            3. Null imputing
            5. Test-train split of the features
        """
        self._label_col = label_col
        self._text_col = text_col
        self._train_size = None
        self._test_size = None
        self._random_state = None

    @property
    def label_col(self):
        return self._label_col

    @property
    def text_col(self):
        return self._text_col
    
    @property
    def meta_features(self):
        return self._meta_features
    
    def load_data(self, staging_tbl, reqd_cols, subsample_perc=100, random_state=1234):
        """
        Public method to reload data from a precreated feature 'staging' table 
        into a pandas dataframe. 
        """
        print('LOADING FEATURE STAGING DATA')
        data = pd.read_csv(staging_tbl)

        if subsample_perc < 100.0:
            data = self._subsample_data(
                data, subsample_perc, random_state
                )
        self._meta_features = [col for col in reqd_cols if col != self.label_col]
        fts = reqd_cols + [self._text_col]

        # GETTING COLS FROM THE TABLE
        return data[fts]

    def _subsample_data(self, data, subsample_perc, random_state):
        return data.sample(frac=subsample_perc/100.0, replace=False, random_state=random_state)

    def binarize_label(self, data, threshold):
        data[self._label_col] = [1 if p > threshold else 0 for p in data[self._label_col]]
        return data 
 
    def create_text_feats(self, text, vect_type, ngram_range=(1,3), max_features=10000):

        if vect_type.lower() == 'tfidf':
            vectorizer = TfidfVectorizer('content', ngram_range=ngram_range, max_features=max_features)
        elif vect_type.lower() == 'bow':
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        
        freq_term_matrix = vectorizer.fit_transform(text)
        self.vocab = vectorizer.get_feature_names()
        
        return pd.DataFrame(freq_term_matrix.toarray(), columns=self.vocab)


    def join_features(self, X, feats):
        return pd.concat([X, feats], axis=1)


    def get_text(self, data):
        return data[self._text_col].astype(str).fillna('')

    def split_X_y(self, data):
        """
        Public method to split dataframe pulled from database into X (features) and y (label)
        """

        y = data[self.label_col]
        X = data[self.meta_features]
        return (X, y)

    def train_test_split(self, X, y, test_size=.2, train_size=.8, random_state=1234, stratify=None): 
        """
        Splits features and labels into train and test splits based on proportion and stratification
        parameters passed by the user. 
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=test_size, 
                                                            random_state=random_state, 
                                                            stratify= stratify)
        if self._train_size == None:
            self._train_size = train_size
        if self._test_size == None:
            self._test_size = test_size
        if self._random_state == None:
            self._random_state = random_state

        return (X_train, X_test, y_train, y_test)


    def impute_nulls(self, data, X_train, X_test):
        """
        impute nulls with zeros
        """
        nas = X_train.isnull().sum()*100/X_train.shape[0]
        na_cols = nas[nas > 0].index.values
        print('COLS TO FILL: {}'.format(str(na_cols)))
        
        for col in na_cols:
            X_train_median = X_train[col].median()
            X_test_median = X_test[col].median()
            print('(X_TRAIN) FILLING NA FOR {} WITH {}'.format(col, X_test_median))
            X_train[col] = X_train[col].fillna(X_train_median)
            X_test[col] = X_test[col].fillna(X_test_median)
            print('(X_TEST) FILLING NA FOR {} WITH {}'.format(col, X_test_median))

        return (X_train, X_test)