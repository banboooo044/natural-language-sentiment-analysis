import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

PATH = "../data/train-val-wakati-juman.tsv"
df = pd.read_table(PATH, index_col=0)
df = df[~pd.isnull(df["text"])]

cv = CountVectorizer(ngram_range=(1,2))
matrix = cv.fit_transform(df["text"])
np.save("./bow_train_x_2-gram.npy", matrix.toarray())
np.save("./bow_train_y_2-gram.npy", np.array(df["label"]))
