import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from src.model import Model
from src.util import Util

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score, f1_score

class ModelMultinomialNB(Model, BaseEstimator, ClassifierMixin):
    def __init__(self, run_fold_name, alpha=1.0, fit_prior=True, class_prior=None):
        params = {
            'alpha' : alpha,
            'fit_prior' : fit_prior,
            'class_prior' : class_prior
        }
        super().__init__(run_fold_name, params)
        self.model = MultinomialNB(**self.params)
        
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        self.model = self.model.fit(tr_x, tr_y)
    
    def fit(self, tr_x, tr_y):
        self.train(tr_x, tr_y)
        return self

    def predict(self, te_x):
        return self.model.predict(te_x)

    def score(self, te_x, te_y):
        y_pred = self.predict(te_x)
        return f1_score(np.identity(5)[te_y], np.identity(5)[y_pred], average='samples')

    def get_params(self, deep=True):
        dic = self.model.get_params(deep)
        dic["run_fold_name"] = self.run_fold_name 
        return dic
    
    def set_params(self, **parameters):
        if "run_fold_name" in parameters:
            self.run_fold_name = parameters["run_fold_name"]
            parameters.pop("run_fold_name", None)
        self.params.update(parameters)
        self.model.set_params(**self.params)
        return self
    
    def save_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)


if __name__ == "__main__":
    params = {
        'alpha' : 1.0,
        'fit_prior' : True,
        'class_prior' : None
    }
    model = ModelMultinomialNB("test", **dict(params))
    check_estimator(model)