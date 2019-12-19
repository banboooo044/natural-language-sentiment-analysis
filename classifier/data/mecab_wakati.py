# MeCabでわかち書き
import MeCab
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

INPUT_PATH = "./corpus-pre.tsv"
# neologd path
DIC_PATH = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"

OUTPUT_PATH = "./corpus-wakati-mecab.tsv"

# 分かち書き ','区切りの,文章を返す
def mecab_list(text):
    tagger = MeCab.Tagger("-Ochasen  -d {0}".format(DIC_PATH))
    tagger.parse('')
    node = tagger.parseToNode(text)
    wakati = []
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        if wclass[0] != u'BOS/EOS':
            if wclass[6] == None or wclass[6] == '*' or wclass[1] =="固有名詞":
                wakati.append(word)
            else:
                wakati.append(wclass[6])
        node = node.next
    return ",".join(wakati)

if __name__ == "__main__":
    df = pd.read_table(INPUT_PATH, index_col=0)
    df["text"] = df["text"].progress_apply(mecab_list)
    df.to_csv(OUTPUT_PATH,sep="\t")