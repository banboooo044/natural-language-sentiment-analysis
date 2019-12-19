import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from src import *
from scipy import sparse

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

from src.util import load_x_train, load_y_train, Logger

logger = Logger()

def makefig(result):
    sns.set_style("whitegrid")
    ax = sns.boxenplot(data = result, width=0.4)
    ax.set_ylabel('Accuracy', size=14)
    ax.tick_params(labelsize=14)
    plt.savefig(f'../model/tuning/{NAME}-logis.png',dpi=300)


if __name__ == '__main__':
    params = {
        'multi_class' :'multinomial', 
        'solver' : 'saga',
        'penalty' : 'l2', 
        'dual' :False, 
        'tol' :0.0001,
        'fit_intercept' : True, 
        'intercept_scaling' : 1, 
        'class_weight' : None, 
        'random_state' : 71, 
        'max_iter' : 500, 
        'verbose' : 1, 
        'warm_start' : False,
        'n_jobs' : None,
        'l1_ratio' : None
    }
    params_logistic = dict(params)
    param_grid_ = { 'C' : [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2 ]  }

    #features = [
    #    "word2vec_mean", "word2vec_max", "word2vec_concat", "word2vec_hier", "fasttext_mean", "fasttext_max", "fasttext_concat", "fasttext_hier"
    #]
    #features = [
    #    "bow", "n-gram","tf-idf", "n-gram-tf-idf"
    #]
    features = [
        'sdv'
    ]

    NAME = ":".join(features)

    results = [ ]
    for name in features:
        x = load_x_train(name)
        y = load_y_train(name)
        model = ModelLogistic(name, **params_logistic)
        search = GridSearchCV( model, cv=6, param_grid=param_grid_ , return_train_score=True )
        search.fit(x, y)
        results.append( (search, name) )
        logger.info(f'{name} - bestscore : {search.best_score_} - result :{search.cv_results_["mean_test_score"]}')
    
    
    res = pd.DataFrame.from_dict(
        { name : search.cv_results_["mean_test_score"] for search, name in results }, 
        orient='index', 
        columns=param_grid_['C']
    )
    
    makefig(res)

    for search, name in results:
        logger.info(f'{name} - bestscore : {search.best_score_}')
    
    res.to_csv(f'../model/tuning/{NAME}-logis.csv')
