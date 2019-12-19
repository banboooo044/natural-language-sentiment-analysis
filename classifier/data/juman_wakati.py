# Jumanでわかち書き
from pyknp import Juman
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
INPUT_PATH = "./corpus-pre.tsv"
# neologd path
DIC_PATH = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"

OUTPUT_PATH = "./corpus-wakati-juman.tsv"

# 分かち書き ','区切りの,文章を返す
def juman_list(text):
    jumanpp = Juman()
    result = jumanpp.analysis(text)
    # アルファベットは全て "En" という文字列に置き換える
    wakati = [ mrph.genkei if mrph.bunrui != "アルファベット" else "En" for mrph in result.mrph_list() ]
    return ",".join(wakati)

if __name__ == "__main__":
    df = pd.read_table(INPUT_PATH, index_col=0)
    df["text"] = df["text"].progress_apply(juman_list)
    df.to_csv(OUTPUT_PATH,sep="\t")
