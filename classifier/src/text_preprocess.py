import torchtext
import MeCab
import re

def remove_stopwords(words, stopwords):
    words = [word for word in words if word not in stopwords]
    return words

def create_stopwords(file_path):
    stop_words = []
    for w in open(file_path, "r"):
        w = w.replace('\n','')
        if len(w) > 0:
            stop_words.append(w)
    return stop_words

# divide to words
def tokenizer_mecab(text):
    path = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"
    m_t = MeCab.Tagger("-Owakati  -d {0}".format(path))
    text = m_t.parse(text)  # これでスペースで単語が区切られる
    ret = text.strip().split()  # スペース部分で区切ったリストに変換
    return ret

# cleaning & normarlize
def preprocessing_text(text):
    # 改行、半角スペース、全角スペースなどを削除
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('　', '', text)
    text = re.sub(' ', '', text)
    text = re.sub('_', '', text)
    text = re.sub('＿', '', text)
    text = re.sub('@', '', text)
    text = re.sub('▁', '', text)

    # 数字文字の一律「0」化
    text = re.sub(r'[0-9 ０-９]', '0', text)  # 数字
    text = text.lower()

    return text

def tokenizer_with_preprocessing(text):
    path = "../data/stopwards.txt"
    stopwords = create_stopwords(path)
    text = preprocessing_text(text)  # 前処理の正規化
    words = tokenizer_mecab(text)  # Mecabの単語分割
    ret = remove_stopwords(words, stopwords) #stopwords
    return ret