import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_logistic import ModelLogistic

# "bow", "bow_nva", "bow_tf-idf", "term_2-gram", "term_3-gram", "word2vec_mean", "word2vec_pre_mean", "word2vec_fine-tuning-iter25", "word2vec_fine-tuning-iter5", "doc2vec"

if __name__ == '__main__':
    params = {
        'multi_class' :'multinomial', 
        'solver' : 'saga',
        'penalty' : 'l2', 
        'dual' :False, 
        'tol' :0.0001,
        'C' : 1000.0, 
        'fit_intercept' : True, 
        'intercept_scaling' : 1, 
        'class_weight' : None, 
        'random_state' : 71, 
        'max_iter' : 500,
        'verbose' : 1,
        'warm_start' : False,
        'n_jobs' : None, 
        'l1_ratio' : None,
    }

    features = [
        "word2vec_mean", "word2vec_max", "word2vec_concat", "word2vec_hier", "fasttext_mean", "fasttext_max", "fasttext_concat", "fasttext_hier"
    ]

    #### Best Parameters
    # bow ->             { 'C' : 0.001 }
    # tf-idf ->          { 'C' : 1.0 }
    # n-gram ->          { 'C' : 1.0 }
    # ngram-tf-idf  ->   { 'C' : 0.1 }
    # word2vec_mean ->   { 'C' : 0.1 }
    # word2vec_max ->    { 'C' : 0.1 }
    # word2vec_concat -> { 'C' : 10.0 }
    # word2vec_hier ->   { 'C' : 0.1 }
    # fasttext_mean ->   { 'C' : 0.001 }
    # fasttex_max ->     { 'C' : 0.001 }
    # fasttext_concat -> { 'C' : 0.001 }
    # fasttext_hier ->   { 'C' : 0.001 }
    # doc2vec-dbow ->    { 'C' : 0.001 }
    # doc2vec-dmpv ->    { 'C' : 0.1   }
    # doc2vec-concat ->  { 'C' : 0.001 }
    # sdv ->             { 'C' : 0.001 }
    # bert ->            { 'C' : 0.001 }

    params['C'] = 0.001
    params_logistic = dict(params)
    runner = Runner(run_name='logis', model_cls=ModelLogistic, features="sdv", params=params_logistic)
    runner.run_train_cv()
    #runner.train_fold(0)
    #runner.run_train_cv([ 100, 500, 1000, 2000, 5000, 6000, 7000, 8000])
    #runner.run_predict_cv()
