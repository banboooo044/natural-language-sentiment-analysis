# データに登場するボキャブラリのみに限定したモデルを作る
import sys
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
from functools import reduce
from keras.preprocessing.text import Tokenizer

class MinifyW2V():
    def __init__(self):
        pass
    
    def load(self, input_path, binary=False):
        """
            モデルのロードを行う
        """
        self.model = KeyedVectors.load_word2vec_format(input_path, binary=binary)
        self.vocab_set = set(self.model.index2word)

    def set_target_vocab(self, target_vocab):
        """
            変更後のボキャブラリを設定
        """
        self.target_vocab = target_vocab

    def save_word2vec(self, output_path, binary=False):
        """
            縮小したボキャブラリでのモデルを保存
        """
        with open(output_path, "wb") as f:
            # write header
            f.write(f"{len(self.target_vocab)} {self.model.vector_size}\n".encode("utf-8"))

            for target_word in self.target_vocab:
                # get vector from model
                if target_word in self.vocab_set:
                    vector = self.model.get_vector(target_word)
                else:
                    vector = np.zeros(self.model.vector_size)

                # save file in a word2vec format
                if binary:
                    target_vector = vector.astype(np.float32).tostring()
                    f.write(target_word.encode("utf-8") + b" " + target_vector)
                else:
                    target_vector = " ".join(repr(val) for val in vector)
                    f.write(f"{target_word} {target_vector}\n".encode("utf-8"))


if __name__ == "__main__":
    # コーパスの読み込み
    PATH = "../data/courpus-wakati-juman.tsv"
    df = pd.read_table(PATH, index_col=0)
    tk = df["text"].apply(lambda x: x.split(","))

    print("build vocab ... ")
    token = Tokenizer(num_words=None)
    token.fit_on_texts(list(tk))
    vocab = token.word_index.keys()
    mw2v = MinifyW2V()
    args = sys.argv
    if args[1] == "word2vec":
        mw2v.load('./word2vec_pre/entity_vector.model.bin', binary=True)
        mw2v.set_target_vocab(target_vocab=list(vocab))
        mw2v.save_word2vec("./word2vec.bin", binary=True)
    elif args[1] == "fasttext":
        mw2v.load('./fasttext_pre/fasttext.vec', binary=False)
        mw2v.set_target_vocab(target_vocab=list(vocab))
        mw2v.save_word2vec("./fasttext.bin", binary=True)
    elif args[1] == "fasttext-neologd":
        mw2v.load('./fasttext_pre/fasttext-neologd.vec', binary=False)
        mw2v.set_target_vocab(target_vocab=list(vocab))
        mw2v.save_word2vec("./fasttext-neologd.bin", binary=True)
    else:
        print("ERROR FILE NOT FOUND")
        exit(1)
