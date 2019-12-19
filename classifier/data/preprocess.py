# テキストの前処理
import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()

INPUT_PATH = "./corpus.tsv"
OUTPUT_PATH = "./corpus-pre.tsv"

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

    # 定型文
    text = re.sub('#いいねした人にやる','',text)
    text = re.sub('\(爆笑\)','',text)
    text = re.sub('\(笑\)','',text)
    text = re.sub('本間', 'ほんま', text)
    # リツイート
    text = re.sub('rt[:]+', '', text)
    # 写真
    text = re.sub('pic.twitter.com/[a-zA-Z0-9]+', '', text)
    # url
    text = re.sub('https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    

    text = re.sub('▁', '', text)
    
    # カタカナ -> ひらがな
    text = "".join([chr(ord(ch) - 96) if ("ァ" <= ch <= "ヴ") else ch for ch in text])
    # ひらがな, カタカナ, 漢字, アルファベット, 数字以外は消す
    text = re.sub('[^ぁ-んァ-ン一-龥ーa-zA-Z0-9\/]+','',text)
    # ww...や00...を1つに置き換え
    text = re.sub('[w]+', 'w', text)
    text = re.sub('[0]+', '0', text)
    
    return text

if __name__ == "__main__":
    df = pd.read_table(INPUT_PATH, index_col=0)
    # ラベルが -1 のものは削除
    df = df[df['label'] != (-1)]
    # 文章が重複してるものは1つにする
    df = df[~df["text"].duplicated()]
    # indexを振り直す
    df.reset_index(drop=True, inplace=True)
    # 前処理を行う
    df["text"] = df["text"].progress_apply(preprocessing_text)
    # 保存
    df.to_csv(OUTPUT_PATH,sep="\t")