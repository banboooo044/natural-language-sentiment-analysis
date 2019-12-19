# 詳しい説明は同様のプログラム logis_gradCV.py を参照
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

from src.runner import Runner
from src.util import Logger
from src.model_NB import ModelMultinomialNB


logger = Logger()

def makefig(result):
    sns.set_style("whitegrid")
    ax = sns.boxenplot(data = result, width=0.4)
    ax.set_ylabel('Accuracy', size=14)
    ax.tick_params(labelsize=14)
    plt.savefig(f'../model/tuning/{NAME}-NB.png',dpi=300)


if __name__ == '__main__':
    base_params = {
        'alpha' : 1.0,
        'fit_prior' : True,
        'class_prior' : None
    }
    params_NB = dict(base_params)
    param_grid_ = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    features = [
        "bow", "n-gram","tf-idf", "n-gram-tf-idf"
    ]

    results = [ ]
    NAME = ":".join(features)
    for name in features:
        x = Runner.load_x_train(name)
        y = Runner.load_y_train()
        model = ModelMultinomialNB(name, **dict(params_NB))
        search = GridSearchCV( model, cv=6, param_grid=param_grid_ , return_train_score=True, verbose=10, refit=True )
        search.fit(x, y)
        results.append( (search, name) )
        logger.info(f'{name} - bestscore : {search.best_score_} - result :{search.cv_results_["mean_test_score"]}')
    
    res = pd.DataFrame.from_dict(
        { name : search.cv_results_["mean_test_score"] for search, name in results }, 
        orient='index', 
        columns=param_grid_['alpha']
    )

    for search, name in results:
        logger.info(f'{name} - bestscore : {search.best_score_}')
    
    res.to_csv(f'../model/tuning/{NAME}-NB.csv')

    makefig(res)
