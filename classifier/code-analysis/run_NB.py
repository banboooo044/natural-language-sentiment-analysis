import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_NB import ModelMultinomialNB

if __name__ == '__main__':
    params = {
        'alpha' : 1.0,
        'fit_prior' : True,
        'class_prior' : None
    }

    #### Best Parameters
    # bow ->             { 'alpha' : 1.0 }
    # tf-tdf ->          { 'alpha' : 1.0 }
    # n-gram ->          { 'alpha' : 1.0 }
    # ngram-tf-idf  ->   { 'alpha' : 0.1 }

    params['alpha'] = 0.1
    params_NB = dict(params)
    runner = Runner(run_name='NB1', model_cls=ModelMultinomialNB, features='n-gram-tf-idf', params=params_NB)
    runner.run_train_cv()
    #runner.run_train_cv([ 100, 500, 1000, 2000, 5000, 6000, 7000, 8000])
    #runner.run_predict_cv()
