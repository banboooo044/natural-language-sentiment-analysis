# Doc2vecを用いて文章分散表現を取得

import os, sys
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

def load_model(path):
    model = KeyedVectors.load(f'./{path}.model')
    return model

def createVector(df, model):
    print("create vector ... ")
    return np.array([ np.array(model.infer_vector(text.split(","))) for idx, (text, _, _) in df.iterrows() ] )

if __name__ == "__main__":
    PATH = "../data/courpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    df = df[~pd.isnull(df["text"])]

    args = sys.argv
    if args[1] == "dbow":
        # https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/ からdbowをダウンロード
        model = KeyedVectors.load("./jawiki.doc2vec.dbow300d/jawiki.doc2vec.dbow300d.model")
        OUTPUT_FILENAME = "doc2vec-dbow"
    elif args[1] == "dmpv":
        # https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/ からdmpvをダウンロード
        model = KeyedVectors.load("./jawiki.doc2vec.dmpv300d/jawiki.doc2vec.dmpv300d.model")
        OUTPUT_FILENAME = "doc2vec-dmpv"
    else:
        model = load_model(args[1])
        OUTPUT_FILENAME = input("[filename].model")

    x = createVector(df, model)
    np.save(f"./{OUTPUT_FILENAME}_x.npy", x)
