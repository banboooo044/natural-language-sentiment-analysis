
import pandas as pd
PATH = "../data/train-val-wakati-juman.tsv"
df = pd.read_table(PATH, index_col=0)
df = df[~pd.isnull(df["text"])]

token = df["text"].apply(lambda x: x.split(","))