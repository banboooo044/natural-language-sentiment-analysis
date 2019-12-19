import sys
sys.path.append('../')
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
import seaborn as sns

from src.util import load_x_train, load_y_train, Logger
from src.model_NB import ModelMultinomialNB


logger = Logger()

def makefig(result):
    sns.set_style("whitegrid")
    ax = sns.boxenplot(data = result, width=0.4)
    ax.set_ylabel('Accuracy', size=14)
    ax.tick_params(labelsize=14)
    plt.savefig(f'../model/tuning/{NAME}-NB.png',dpi=300)


if __name__ == '__main__':
    params = {
        'alpha' : 1.0,
        'fit_prior' : True,
        'class_prior' : None
    }
    params_NB = dict(params)
    param_grid_ = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    #features = [
    #    "bow","bow_nva","bow_tf-idf","term_2-gram","term_3-gram","word2vec_mean","word2vec_pre_mean",
    #    "word2vec_fine-tuning", "doc2vec", "scdv", "bert"
    #]
    features = [
        "bow", "n-gram","tf-idf", "n-gram-tf-idf"
    ]
    #mll_scorer = make_scorer(lambda x,y: f1_score(x,y ,average='micro'), greater_is_better=False)
    results = [ ]
    NAME = ":".join(features)
    for name in features:
        x = load_x_train(name)
        y = load_y_train(name)
        model = ModelMultinomialNB(name, **dict(params))
        search = GridSearchCV( model, cv=6, param_grid=param_grid_ , return_train_score=True, verbose=10, refit=True )
        search.fit(x, y)
        results.append( (search, name) )
        logger.info(f'{name} - bestscore : {search.best_score_} - result :{search.cv_results_["mean_test_score"]}')
    
    res = pd.DataFrame.from_dict(
        { name : search.cv_results_["mean_test_score"] for search, name in results }, 
        orient='index', 
        columns=param_grid_['alpha']
    )
    
    makefig(res)

    for search, name in results:
        logger.info(f'{name} - bestscore : {search.best_score_}')
    
    res.to_csv(f'../model/tuning/{NAME}-NB.csv')
