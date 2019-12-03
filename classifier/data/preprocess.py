import MeCab
import pandas as pd
import re
from pyknp import Juman
from tqdm import tqdm

tqdm.pandas()

INPUT_PATH = "/Users/banboooo044/Documents/natural-language-sentiment-anaysis/classifier/data/train-val-small.tsv"
OUTPUT_PATH = "/Users/banboooo044/Documents/natural-language-sentiment-anaysis/classifier/data/train-val-pre.tsv"
DIC_PATH = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"

# cleaning & normarlize
def preprocessing_text(text):
    # 改行、半角スペース、全角スペースなどを削除
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('　', '', text)
    text = re.sub(' ', '', text)
    # 名前
    text = re.sub('@[a-zA-Z_0-9]+','',text)
    # ハッシュタグ
    text = re.sub('#[^▁]+','',text)
    text = re.sub('_', '', text)
    text = re.sub('＿', '', text)
    
    # 数字文字の一律「0」化
    text = re.sub(r'[0-9 ０-９]', '0', text)  # 数字
    text = text.lower()

    # ケース
    text = re.sub('#いいねした人にやる','',text)
    # リツイート
    text = re.sub('rt[:]+', '', text)
    # 写真
    text = re.sub('pic.twitter.com/[a-zA-Z0-9]+', '', text)
    # url
    text = re.sub('https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    text = re.sub('\(爆笑\)','',text)
    text = re.sub('\(笑\)','',text)
    #text = re.sub('[ー]+', 'ー', text)

    text = re.sub('▁', '', text)
    text = re.sub('本間', 'ほんま', text)
    # ひらがなカタカナ漢字数字
    text = re.sub('[^ぁ-んァ-ン一-龥ーa-zA-Z0-9\/]+','',text)
    text = re.sub('[w]+', 'w', text)
    text = re.sub('[0]+', '0', text)

    return text

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

def mecab_extract_nva(text):
    tagger = MeCab.Tagger("-Ochasen  -d {0}".format(DIC_PATH))
    tagger.parse('')
    node = tagger.parseToNode(text)
    wakati = []
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        if wclass[0] != u'BOS/EOS' and (wclass[0] == "名詞" or wclass[0] == "動詞" or wclass[0] == "形容詞"):
            if wclass[6] == None or wclass[6] == '*' or wclass[1] =="固有名詞":
                wakati.append(word)
            else:
                wakati.append(wclass[6])
        node = node.next
    return ",".join(wakati)

def juman_list(text):
    jumanpp = Juman()
    result = jumanpp.analysis(text)
    wakati = [ mrph.genkei for mrph in result.mrph_list() ]
    return ",".join(wakati)

def juman_extract_nva(text):
    jumanpp = Juman()
    result = jumanpp.analysis(text)
    wakati = [ mrph.genkei for mrph in result.mrph_list() if mrph.hinsi == "名詞" or mrph.hinsi == "動詞" or mrph.hinsi == "形容詞" ]
    return ",".join(wakati)

if __name__ == "__main__":
    df = pd.read_table(INPUT_PATH, index_col=0)
    df = df[df['label'] != (-1)]
    df = df[~df["text"].duplicated()]
    df.reset_index(drop=True, inplace=True)
    df["text"] = df["text"].progress_apply(preprocessing_text)
    #df["text"] = df["text"].progress_apply(juman_extract_nva)
    df.to_csv(OUTPUT_PATH,sep="\t")
