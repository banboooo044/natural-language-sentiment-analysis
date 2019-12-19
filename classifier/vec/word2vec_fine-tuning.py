# fine-tuning Word2Vec model and save model
import pandas as pd
import logging
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.preprocessing import normalize
import sys
import re

class EpochLogger(CallbackAny2Vec):
    def __init__(self, path_prefix):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

# 追加学習を行う
def additional_train(token):
    print("build model ... ")
    epoch_logger = EpochLogger()
    # 追加学習を行う.
    model = Word2Vec(workers=40, hs = 0, sg = 1, negative = 10, iter = 5,\
            size=200, min_count=1, \
            window=10, sample=1e-3, seed=1, compute_loss=True, callbacks=[epoch_logger])
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

if __name__ == "__main__":
    PATH = "../data/courpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]
    token = df["text"].apply(lambda x: x.split(","))

    OUTPUT_FILENAME = "word2vec-fine-tuning"
    model = additional_train(token)
    save_model(model, OUTPUT_FILENAME)
