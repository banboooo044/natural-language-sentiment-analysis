import os,sys
sys.path.append('../')

import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.model import Model
from src.util import Util

from sklearn.metrics import log_loss, accuracy_score, f1_score, classification_report

class ModelLGB(Model):
    def __init__(self, run_fold_name, **params):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        validation = va_x is not None
        dtrain = lgb.Dataset(tr_x, label=tr_y)
        if validation:
            dvalid = lgb.Dataset(va_x, label=va_y)

        params = dict(self.params)
        num_round = params.pop('num_boost_round')

        if validation:
            # バリデーションデータが存在する場合, Eearly Stoppingを行う
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [dtrain, dvalid ]
            self.model = lgb.train(params, dtrain, num_round, valid_sets=watchlist,
                                valid_names=['train','eval'],
                                early_stopping_rounds=early_stopping_rounds)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = lgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, te_x):
        return self.model.predict(te_x, ntree_limit=self.model.best_iteration)

    def score(self, te_x, te_y):
        pred_prob = self.predict(te_x)
        y_pred = np.argmax(pred_prob, axis=1)
        # print(classification_report(te_y, y_pred))
        return f1_score(np.identity(5)[te_y], np.identity(5)[y_pred], average='samples')

    def save_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)