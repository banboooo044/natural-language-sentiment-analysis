import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from src.model_lgb import ModelLGB
from src.runner import Runner

# "bow", "bow_nva", "bow_tf-idf", "term_2-gram", "term_3-gram", "word2vec_mean", "word2vec_pre_mean", "word2vec_fine-tuning-iter25", "word2vec_fine-tuning-iter5", "doc2vec"

if __name__ == '__main__':
    params_xgb = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbose_eval': True,
        'num_class' : 5,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_boost_round': 10000,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_boost_round'] = 350
    # xgboostによる学習・予測
    runner = Runner('lgb1', ModelLGB, "bow", params_xgb)
    runner.run_train_cv()
    #runner.run_predict_cv()
    