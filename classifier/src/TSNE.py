import sklearn.base
import bhtsne
import numpy as np


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, n_components=2, perplexity=30.0, theta=0.5, random_state=-1):
        self.n_components = n_components
        self.perplexity = perplexity
        self.theta = theta
        self.random_state = random_state

    def fit_transform(self, x):
        return bhtsne.tsne(
            x.astype(np.float64), dimensions=self.n_components, perplexity=self.perplexity, theta=self.theta,
            rand_seed=self.random_state)
    
