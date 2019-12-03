import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from src import *
from scipy import sparse

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

logger = Logger()

def makefig(result):
    sns.set_style("whitegrid")
    ax = sns.boxenplot(data = result, width=0.4)
    ax.set_ylabel('Accuracy', size=14)
    ax.tick_params(labelsize=14)
    plt.savefig(f'../model/tuning/{name}-logis.png',dpi=300)

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
        'max_iter' : 100, 
        'verbose' : 0, 
        'warm_start' : False,
        'n_jobs' : None, 
        'l1_ratio' : None
    }
    params_logistic = dict(params)
    param_grid_ = { 'C' : [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2 ]  }

    PATH = [
        ('../vec/bow_train_x.npy', '../vec/bow_train_y.npy', "bow"),
        ('../vec/bow_train_x_nva.npy', '../vec/bow_train_y_nva.npy', "bow_nva"),
        ('../vec/bow_train_x_tf-idf.npy', '../vec/bow_train_y_tf-idf.npy', "bow_tf-idf"),
        ('../vec/bow_train_x_2-gram.npy', '../vec/bow_train_y_2-gram.npy', "term_2-gram"),
    ]

    results = [ ]
    for x_path, y_path, name in PATH:
        x = sparse.csr_matrix(np.load(x_path, allow_pickle = True), dtype=np.float64)
        y = np.load(y_path, allow_pickle = True)
        model = ModelLogistic(name, **params_logistic)
        search = GridSearchCV( model, cv=6, param_grid=param_grid_ , return_train_score=True )
        search.fit(x, y)
        results.append( (search, name) )
    
    res = pd.DataFrame.from_dict(
        { name : search.cv_results_["mean_test_score"] for search, name in results }, 
        orient='index', 
        columns=param_grid_['C']
    )
    
    makefig(res)

    for search, name in results:
        logger.info(f'{name} - bestscore : {search.best_score_}')
    
    res.to_csv(f'../model/tuning/{name}-logis.csv')
