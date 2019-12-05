import os,sys
sys.path.append('../')

import os

import numpy as np
import pandas as pd
import xgboost as xgb

from src.model import Model
from src.util import Util

from sklearn.metrics import log_loss, accuracy_score

class ModelXGB(Model):
    def __init__(self, run_fold_name, **params):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')

        # 学習
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
                                early_stopping_rounds=early_stopping_rounds)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def score(self, te_x, te_y):
        dtest = xgb.DMatrix(te_x)
        pred_prob = self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)
        return accuracy_score(te_y, np.argmax(pred_prob, axis=1))

        ## 2 class
        #pred_prob[pred_prob > 0.5] = 1
        #pred_prob[pred_prob <= 0.5] = 0
        #return accuracy_score(te_y, pred_prob)

    def save_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)