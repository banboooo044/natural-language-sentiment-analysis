import pandas as pd
df = pd.read_table("./train-val.tsv", index_col=0)
idx = int(input())
print(df.iat[idx, 0])
label = int(input())
df.iat[idx, 1] = label
df.to_csv("./train-val.tsv",sep="\t")