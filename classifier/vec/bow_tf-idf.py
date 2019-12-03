import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

PATH = "../data/train-val-wakati-juman.tsv"
df = pd.read_table(PATH, index_col=0)
df = df[~pd.isnull(df["text"])]

Tfidf_cv = TfidfVectorizer()
matrix =  Tfidf_cv.fit_transform(df["text"])

np.save("./bow_train_x_tf-idf.npy", matrix.toarray())
np.save("./bow_train_y_tf-idf.npy", np.array(df["label"]))
