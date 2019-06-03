from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, accuracy_score, \
    f1_score, precision_recall_curve, make_scorer, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from datetime import datetime as dt
import logging
import sys
import os
import joblib

class Pipeline:
    def __init__(self, pipeline_mode, grid_model_id_key=None, 
                 X_train=None, y_train=None,
                 clf_grid=None, model_obj_pref='', scoring='accuracy', 
                 threshold=0.5, model_obj_path=''):      
        """
        The pipeline class is used for the following three functions:
            1. Find the optimal classifier to run given a hyperparameter 
               grid (build mode)
            2. Refresh a pre-determined classifier with new data (refresh mode)
            3. Load a pre-trained classifier and predict on new data (load mode)

        The pipeline class contains methods to calculate accuracy, recall, 
        precision, and f1 evaluation metrics at a user-specified threshold.

        Running the pipeline in build mode also a generates a joblib dump of 
        the model.
        
        :param pipeline_mode: The type of pipeline object to be initialised. Can
                              take options as "build", "refresh" and "load", 
                              Class description explains these different modes.
        :type pipeline_mode: string
        :param grid_model_id_key: The key corresponding to the model class to 
                                  be used from the input dictionary, defaults to
                                  None
        :type grid_model_id_key: string, optional
        :param X_train: Training regressor dataset to train classifier, must 
                        follow format required by scikit models. Used only in
                        build or refresh mode, defaults to None
        :type X_train: pandas df/numpy array, optional
        :param y_train: Training dependent dataset to train classifier, must 
                        follow format required by scikit models. Used only in
                        build or refresh mode, defaults to None 
        :type y_train: pandas series/numpy array, optional
        :param clf_grid: Hyperparameter grid to search over while building the new model
        :type clf_grid: Dictionary, optional
        :param model_obj_path: Path of the pickled sklearn pipeline object
                               generated in a previous run, defaults to None
        :type model_obj_path: string, optional
        :param model_obj_pref: Prefix to be appended to the name of the model
                               pickle dump, defaults to an empty string
        :type model_obj_pref: string, optional
        :param scoring: Scoring measures to be used by the optimizer. Choice of 
                        "precision" or "recall". Calculates selected metric at
                        threshold defined by parameter. Defaults to "recall"
        :type scoring: string, optional
        :param threshold: Threshold number of prediction at which scoring metric
                          needs to be calculated
        :type threshold: int, optional
        """
        self.X_train = X_train
        self.y_train = y_train
        self.clf_grid = clf_grid 
        self.model_obj_pref = model_obj_pref 
        self.scoring = scoring
        self.threshold = threshold
      

        if pipeline_mode.lower() == 'build':
            scorer = self._make_score_fxn()
            print('USING SCORING FUNCTION: {}'.format(scorer))
            grid_obj = self._train_grid(scorer= scorer, key=grid_model_id_key)
            self._estimator = grid_obj.best_estimator_
        elif pipeline_mode.lower() == 'refresh':
            self._estimator = self._model_refresh(model_obj_path = model_obj_path,
                                                  X_train = X_train, 
                                                  y_train = y_train)
        elif pipeline_mode.lower() == 'load':
            self._estimator = self._load_model_obj(model_obj_path = model_obj_path)

    @property
    def estimator(self):
        """
        the classifier
        """
        return self._estimator

    def _make_score_fxn(self):
        """
        Private wrapper function to either create a precision or recall scorer
        function to be used in the model training process

        :return: sklearn scorer object that returns a scalar score; greater is better
        :rtype: scorer object
        """

        if self.scoring == 'accuracy':
            return make_scorer(self.accuracy_at_k)
        elif self.scoring == 'precision':
            return make_scorer(self.precision_at_k)
        elif self.scoring == 'recall':            
            return make_scorer(self.recall_at_k)
        elif self.scoring == 'auc-roc':
            return make_scorer(self.auc_roc)
        else: # f1 score
            return make_scorer(self.f1_at_k)

    def _train_grid(self, scorer, key):       
        """
        :param scorer: Sklearn scorer object that returns a scalar precision or recall 
                    score; greater is better
        :type scorer: scorer object
        :param key: key corresponding to classifier in grid (eg. 'DT' = 'Decision Tree')
        :type key: string
        :return: trained classifier representing best model (based on metric used by scorer)
        :rtype: sklearn classifier 
        """
        model = self.clf_grid[key]['type']
        parameters = self.clf_grid[key]['grid']

        print(model, parameters)
        clf = GridSearchCV(model, parameters, scoring=scorer, cv=5, return_train_score=True)

        clf.fit(self.X_train, self.y_train)
        time_now = dt.now()
        filepath_base = 'models_store'

        print(model)
        print(clf)
        print(clf.best_estimator_)
        print(clf.best_score_)
        print(filepath_base)
        print(os.getcwd())
        
        cv_results_file_name = "{}_{}_{}results.csv".format(self.model_obj_pref, key, time_now)
        filepath = os.path.join(filepath_base, cv_results_file_name)
        df = pd.DataFrame(clf.cv_results_)
        if not os.path.exists('models_store'):
            print('PATH DOES NOT EXIST, CREATING DIRECTORY {}'.format(filepath_base))
            os.mkdir(filepath_base)
        try:
            df.to_csv(filepath)
        except Exception as e:
            print(e)
            return
        
        model_obj_file_name = '{}_{}_{}.joblib'.format(self.model_obj_pref, key, time_now)
        filepath = os.path.join(filepath_base, model_obj_file_name)
        joblib.dump(clf.best_estimator_ , filepath)
        print('MODEL STORED AT {}'.format(filepath))
        
        return clf

    def _model_refresh(self, model_obj_path, X_train, y_train):
        """
        Private function to execute the "model refresh" pipeline. Activities
        in this pipeline are:
            1. Load a pre-built pipeline
            2. Train the pipeline
        """

        classifier_obj = self._load_model_obj(model_obj_path)
        classifier_obj = classifier_obj.fit(X_train, y_train)

        time_now = dt.now()      
        filepath_base = 'models_store'
        model_obj_file_name = '{}_{}.joblib'.format(self.model_obj_pref, time_now)
        filepath = os.path.join(filepath_base, model_obj_file_name)
        joblib.dump(classifier_obj, filepath)

        return classifier_obj 

    def _load_model_obj(self, model_obj_path):
        """
        Private function to load a model joblib dump. 
        """
        filepath_base = 'models_store'
        filepath = os.path.join(filepath_base, model_obj_path)
        print(filepath)

        return joblib.load(filepath)

    def gen_pred_probs(self, X_test):
        """
        Generates predicted probabilities of class membership
        """
        return self.estimator.predict_proba(X_test)[:, 1]


    ##### Model Evaluation Functions #####
    def generate_binary_at_k(self, y_pred_probs, k):
        """
        Turn probabilistic outcomes to binary variable 
        based on probability threshold k. 
        """
        return [1 if p > k else 0 for p in y_pred_probs]

    
    def accuracy_at_k(self, y_test, y_pred_probs, k):
        """
        Calculate precision of predictions at a threshold k.
        """

        preds_at_k = self.generate_binary_at_k(y_pred_probs, k)
        accuracy = accuracy_score(y_test, preds_at_k)

        return accuracy

    def precision_at_k(self, y_test, y_pred_probs, k):
        """
        Calculate precision of predictions at a threshold k.
        """

        preds_at_k = self.generate_binary_at_k(y_pred_probs, k)
        precision = precision_score(y_test, preds_at_k)

        return precision

    def recall_at_k(self, y_test, y_pred_probs, k):
        """
        Calculate recall of predictions at a threshold k.
        """

        preds_at_k = self.generate_binary_at_k(y_pred_probs, k)
        recall = recall_score(y_test, preds_at_k)

        return recall

    def f1_at_k(self, y_test, y_pred_probs, k):
        """
        Calculate F1 score of predictions at a threshold k.
        """
        preds_at_k = self.generate_binary_at_k(y_pred_probs, k)
        f1 = f1_score(y_test, preds_at_k)
        return f1
    
    def auc_roc(self, y_test, y_pred_probs):
        return roc_auc_score(y_test, y_pred_probs)
    
    def confusion_matrix(self, y_test, y_pred_probs, k):
        preds_at_k = self.generate_binary_at_k(y_pred_probs, k)
        return pd.DataFrame(confusion_matrix(y_test, preds_at_k))
    
    def feature_importance(self, top_k):
        '''
        identify important features using a random forest
        '''

        importances = self._estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = self.X_train.columns.values
        # Feature Ranking
        importance = pd.DataFrame(columns=['feature', 'importance'])
        for f in range(0, top_k):
            importance.loc[f+1] = [features[indices[f]], importances[indices[f]]]

        return importance

    
def model_exec(run_type, iteration_name, grid, 
                scoring, score_k_val, 
                X_train, y_train, X_test, y_test):
    
    model_obj_path = 'models_store'
    # Logging the modeling parameters
    log_mdl_msg = ('The path of the model object used for this run is found at \n{}\n' +
                   'The scoring function used for this run is: {}\n' +
                   'The probabilistic threshold at which the score is calculated is {}\n' +
                   'The parameter grid used for the search is: \n{}\n').format(
                       model_obj_path, scoring, score_k_val, grid
                       )
    print(log_mdl_msg)

    print('BEGINNING GRID SEARCH')
    for key in grid.keys():
        try:
            pipeline = Pipeline(
                pipeline_mode=run_type, 
                grid_model_id_key=key, 
                scoring=scoring,
                X_train=X_train, 
                y_train=y_train, 
                clf_grid=grid, 
                threshold=score_k_val, 
                model_obj_path=model_obj_path, 
                model_obj_pref=iteration_name
            )

            print('TESTING SELECTED MODEL ON TEST DATA')
            y_test_prob = pipeline.gen_pred_probs(X_test)
            recall = pipeline.recall_at_k(y_test, y_test_prob, score_k_val)
            precision = pipeline.precision_at_k(y_test, y_test_prob, score_k_val) 
            accuracy = pipeline.accuracy_at_k(y_test, y_test_prob, score_k_val)
            cm = pipeline.confusion_matrix(y_test, y_test_prob, score_k_val)

            print('TEST PRECISION AT {}: {}'.format(score_k_val, precision))
            print('TEST RECALL AT {}: {}'.format(score_k_val, recall))
            print('TEST ACCURACY AT {}: {}'.format(score_k_val, accuracy))
            print('CONFUSION MATRIX:')
            print(cm)

        except Exception as e:
            print('MODEL BUILD FAILED: {}'.format(e))
