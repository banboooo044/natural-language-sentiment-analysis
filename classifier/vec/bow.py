import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import numpy as np

PATH = "../data/train-val-wakati-juman-nva.tsv"
df = pd.read_table(PATH, index_col=0)
df = df[~pd.isnull(df["text"])]

cv = CountVectorizer()
matrix = cv.fit_transform(df["text"])

np.save("./bow_train_x_nva.npy", matrix.toarray())
np.save("./bow_train_y_nva.npy", np.array(df["label"]))
