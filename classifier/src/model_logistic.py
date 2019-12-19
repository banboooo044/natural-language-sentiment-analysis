import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

from src.model import Model
from src.util import Util

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, classification_report,  precision_recall_curve

from scipy.sparse import issparse

class ModelLogistic(Model, BaseEstimator, ClassifierMixin):
    def __init__(self, run_fold_name, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                    intercept_scaling=1, class_weight=None, random_state=None, solver='warn', max_iter=100, 
                        multi_class='warn', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        params = {
            'multi_class' : multi_class,
            'solver' : solver,
            'penalty' : penalty,
            'dual' : dual,
            'tol' : tol,
            'C' : C,
            'fit_intercept' : fit_intercept,
            'intercept_scaling' : intercept_scaling,
            'class_weight' : class_weight,
            'random_state' : random_state,
            'max_iter' : max_iter,
            'verbose' : verbose,
            'warm_start' : warm_start,
            'n_jobs' : n_jobs,
            'l1_ratio' : l1_ratio
        }
        super().__init__(run_fold_name, params)
        self.model = LogisticRegression(**self.params)
        
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # 標準化 (x - mu) / sigma
        if issparse(tr_x):
            # スパース行列の時は mu = 0 で近似
            scaler = StandardScaler(with_mean=False)
        else:
            scaler = StandardScaler()
        scaler_sc = scaler.fit(tr_x)
        tr_x = scaler_sc.transform(tr_x)

        # モデルで学習
        self.model = self.model.fit(tr_x, tr_y)
        self.scaler = scaler
    
    def fit(self, tr_x, tr_y):
        self.train(tr_x, tr_y)
        return self

    def predict(self, te_x):
        te_x = self.scaler.transform(te_x)
        return self.model.predict(te_x)

    def score(self, te_x, te_y):
        y_pred = self.predict(te_x)
        #print(classification_report(te_y, y_pred))
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
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)
        self.scaler = Util.load(scaler_path)
