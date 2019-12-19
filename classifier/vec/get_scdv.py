
# FasttextのSCDVを取得
import sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
import logging
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import normalize
import re

from src.SCDV import SCDVVectorizer

def save_model(model, filename):
    print("save model ...")
    model.save(f"./{filename}.model")

def load_model(filename):
    print("load model ... ")
    model = Word2Vec.load(f"./{filename}.model")
    return model

if __name__ == "__main__":
    PATH = "../data/courpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    token = df["text"].apply(lambda x: x.split(","))

    gmm_params = {'n_components' : 30,'covariance_type' : 'tied', 'init_params' : 'kmeans', 'max_iter' : 50 , 'random_state' : 71, 'verbose' : 2} 
    vt = SCDVVectorizer(embedding_size = 300, sparsity_percentage = 0.05, 
                    gaussian_mixture_parameters=gmm_params, embedding_array = None)
    vec = vt.fit_transform(token, l2_normalize=True)

    OUTPUT_FILENAME = "fasttext_scdv"
    np.save(f"./{OUTPUT_FILENAME}_x.npy", vec)
    