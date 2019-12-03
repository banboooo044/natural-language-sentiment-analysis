import pandas as pd

# "喜び" : 0, "悲しみ" : 1, "怒り" : 2, "恐怖" : 3, "その他" : 4
df = pd.read_table("./train-val.tsv", index_col=0)
label = list(df["label"])
update = list(df["update"])
for idx, (lab, text, upd) in df.iterrows():
    if upd:
        continue

    print("************************************:")
    print("喜び : 0, 悲しみ : 1, 怒り : 2, 恐怖 : 3, その他 : 4")
    print("INDEX : {0} / {1}".format(idx, len(df)))
    print("************************************:")
    print("LABEL : {0}".format(lab))
    print("************************************:")
    print(text)
    print("************************************:")
    print(">>>",end="")
    try:
        label[idx] = int(input())
        update[idx] = True
    except KeyboardInterrupt:
        break
    except ValueError:
        break

df["label"] = pd.Series(label)
df["update"] = pd.Series(update)

df.to_csv("./train-val.tsv",sep="\t")
