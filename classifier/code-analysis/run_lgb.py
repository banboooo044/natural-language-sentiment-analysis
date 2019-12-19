import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from src.model_lgb import ModelLGB
from src.runner import Runner

# "bow", "bow_nva", "bow_tf-idf", "term_2-gram", "term_3-gram", "word2vec_mean", "word2vec_pre_mean", "word2vec_fine-tuning-iter25", "word2vec_fine-tuning-iter5", "doc2vec"

if __name__ == '__main__':
    params_lgb = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class' : 5,
        'silent': 1,
        'random_state': 71,
        'num_boost_round': 10000,
        'early_stopping_rounds': 10,
        'n_estimator' : 500
    }

    #### Best Parameters
    # bow ->             { 'num_leaves' : 32, 'colsample_bytree' : 0.466 }
    # tf-idf ->          { 'num_leaves' : 22, 'colsample_bytree' : 0.540 }
    # n-gram ->          { 'num_leaves' : 34, 'colsample_bytree' : 0.689 }
    # ngram-tf-idf  ->   { 'num_leaves' : 26, 'colsample_bytree' : 0.393 }
    # word2vec_mean ->   { 'num_leaves' : 20, 'colsample_bytree' : 0.379 }
    # word2vec_max ->    { 'num_leaves' : 22, 'colsample_bytree' : 0.387 }
    # word2vec_concat -> { 'num_leaves' : 16, 'colsample_bytree' : 0.310 }
    # word2vec_hier ->   { 'num_leaves' : 30, 'colsample_bytree' : 0.888 }
    # fasttext_mean ->   { 'num_leaves' : 34, 'colsample_bytree' : 0.546, 'subsample' : 0.7725, 'learning_rate': 0.01 }
    # fasttex_max ->     { 'num_leaves' : 28, 'colsample_bytree' : 0.447 }
    # fasttext_concat -> { 'num_leaves' : 12, 'colsample_bytree' : 0.344 }
    # fasttext_hier ->   { 'num_leaves' : 10, 'colsample_bytree' : 0.319 }
    # doc2vec-dbow ->    { 'num_leaves' : 46, 'colsample_bytree' : 0.303, 'subsample' : 0.879, 'learning_rate': 0.01 }
    # doc2vec-dmpv ->    { 'num_leaves' : 30, 'colsample_bytree' : 0.597, 'subsample' : 0.910, 'learning_rate': 0.01 }
    # doc2vec-concat ->  { 'num_leaves' : 25, 'colsample_bytree' : 0.624, 'subsample' : 0.590, 'learning_rate': 0.05 }
    # sdv ->             {'colsample_bytree': '0.539', 'learning_rate': 0.01, 'num_leaves': 56, 'subsample': 0.942}
    # bert ->            { 'num_leaves' : 24, 'colsample_bytree' : 0.336, 'subsample' : 0.990, 'learning_rate': 0.05 }

    params_lgb.update({'colsample_bytree': '0.539', 'learning_rate': 0.01, 'num_leaves': 56, 'subsample': 0.942})
    params_lgb_all = dict(params_lgb)
    # xgboostによる学習・予測
    runner = Runner('lgb1', ModelLGB, "sdv", params_lgb_all)
    
    runner.run_train_cv()
    #runner.run_predict_cv()
    