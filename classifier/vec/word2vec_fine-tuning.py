import pandas as pd
import logging
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import normalize
import sys
import re

def additional_train(token, num_features=200, min_word=10, num_worker=40, window=10, downsampling=1e-3):
    print("build model ... ")
    model = Word2Vec(workers=num_worker, hs = 0, sg = 1, negative = 10, iter = 5,\
            size=num_features, min_count = min_word, \
            window = window, sample =downsampling, seed=1, compute_loss = True)
    model.build_vocab(token)
    total_examples = model.corpus_count
    pre_model = KeyedVectors.load_word2vec_format('../data/entity_vector/entity_vector.model.bin', binary=True)
    model.build_vocab([list(pre_model.vocab.keys())], update=True)
    model.intersect_word2vec_format('../data/entity_vector/entity_vector.model.bin', binary=True)

    print("train ... ")
    model.train(token, total_examples=total_examples, epochs=model.iter)
    return model
    
def save_model(model, filename):
    print("save model ...")
    model.save(f"./{filename}.model")

def load_model(filename):
    print("load model ... ")
    model = Word2Vec.load(f"./{filename}.model")
    return model

def createVector(df, model, num_features=200):
    print("create vector ... ")
    x = np.empty((0, num_features))
    for idx, (text, label, _) in df.iterrows():
        tmp = np.array([ np.array(model.wv[word])  for word in text.split(",") if word in model.wv.vocab ])
        if len(tmp) != 0:
            x = np.append(x, np.mean(tmp, axis=0).reshape(1, num_features), axis=0)
        else:
            x = np.append(x, np.zeros((1, num_features)), axis=0)
    return x

if __name__ == "__main__":
    # f = "word2vec_fine-tuning-iter25" / "word2vec_fine-tuning-iter5"
    f = "word2vec_fine-tuning-iter5-tmp"
    PATH = "../data/train-val-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    token = df["text"].apply(lambda x: x.split(","))

    # if you want to train
    #model = additional_train(token, min_word=1)
    #save_model(model, f)

    # if you want to create embedding vector
    model = load_model(f)
    x = createVector(df, model)
    np.save(f"./{f}_x.npy", x)