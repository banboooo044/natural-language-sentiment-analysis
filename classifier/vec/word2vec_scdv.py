import sys, os
sys.path.append('../')

import pandas as pd
import logging
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import normalize
import sys
import re

from src import *
    
def save_model(model, filename):
    print("save model ...")
    model.save(f"./{filename}.model")

def load_model(filename):
    print("load model ... ")
    model = Word2Vec.load(f"./{filename}.model")
    return model

if __name__ == "__main__":
    # f = "word2vec_fine-tuning-iter25" / "word2vec_fine-tuning-iter5"
    f = "word2vec_scdv"
    PATH = "../data/train-val-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    token = df["text"].apply(lambda x: x.split(","))

    embedding_array = np.load('../vec/word2vec_fine-tuning-iter5_x.npy')

    gmm_params = { 'n_components' : 60,'covariance_type' : 'tied', 'init_params' : 'kmeans', 'max_iter' : 50 , 'random_state' : 71, 'verbose' : 2} 
    vt = SCDVVectorizer(embedding_size = 200, sparsity_percentage = 0.04, 
                    gaussian_mixture_parameters=gmm_params, embedding_array = None)
    
    vec = vt.fit_transform(token, l2_normalize=True)

    np.save(f"./{f}_x.npy", vec)
    