import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from src.model_xgb import ModelXGB
from src.runner import Runner

# "bow", "bow_nva", "bow_tf-idf", "term_2-gram", "term_3-gram", "word2vec_mean", "word2vec_pre_mean", "word2vec_fine-tuning-iter25", "word2vec_fine-tuning-iter5", "doc2vec"

if __name__ == '__main__':
    params_xgb = {
        #'objective': 'multi:softprob',
        'objective' : 'binary:logistic',
        'eval_metric': 'logloss',
        'verbose_eval': True,
        #'num_class': 2,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10000,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 350
    # xgboostによる学習・予測
    runner = Runner('xgb1', ModelXGB, "term_3-gram", params_xgb)
    runner.run_train_cv()
    #runner.run_predict_cv()
    