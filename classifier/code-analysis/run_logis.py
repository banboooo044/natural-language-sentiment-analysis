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
        'C' : 1.0, 
        'fit_intercept' : True, 
        'intercept_scaling' : 1, 
        'class_weight' : None, 
        'random_state' : 71, 
        'max_iter' : 500,
        'verbose' : 0,
        'warm_start' : False,
        'n_jobs' : None, 
        'l1_ratio' : None
    }
    params_logistic = dict(params)
    runner = Runner(run_name='logis', model_cls=ModelLogistic, features="", params=params_logistic)
    runner.run_train_cv()
    #runner.run_train_cv([ 100, 500, 1000, 2000, 5000, 6000, 7000, 8000])
    #runner.run_predict_cv()
