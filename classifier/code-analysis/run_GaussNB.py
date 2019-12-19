import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from src.runner import Runner
from src.model_GaussNB import ModelGaussNB

# "bow", "bow_nva", "bow_tf-idf", "term_2-gram", "term_3-gram", "word2vec_mean", "word2vec_pre_mean", "word2vec_fine-tuning-iter25", "word2vec_fine-tuning-iter5", "doc2vec"

if __name__ == '__main__':
    params = {
        'priors' : None, 
        'var_smoothing' : 1e-09
    }
    features = [
        "word2vec_mean", "word2vec_max", "word2vec_concat", "word2vec_hier", "fasttext_mean", "fasttext_max", "fasttext_concat", "fasttext_hier"
    ]
    params_NB = dict(params)
    runner = Runner(run_name='GNB1', model_cls=ModelGaussNB, features='bert', params=params_NB)
    runner.run_train_cv()
    #runner.run_train_cv([ 100, 500, 1000, 2000, 5000, 6000, 7000, 8000])
    #runner.run_predict_cv()
