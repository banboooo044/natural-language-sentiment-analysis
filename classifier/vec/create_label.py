import pandas as pd
import numpy as np

# 分かち書きされたデータを読み込む
PATH = "../data/courpus-wakati-juman.tsv"
df = pd.read_table(PATH, index_col=0)
df = df[~pd.isnull(df["text"])]

# 保存
np.save('./y_full.npy', np.array(df["label"]))