# natural-language-sentiment-analysis
元データ : data/corpus.tsv
## 学習済みモデルのダウンロード
* Word2vec
「東北大学 乾・岡崎研究室の公開モデル」
  - http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
  - 学習データ: 日本語Wikipedia
  - 200次元

  置き場所: classifier/vec/word2vec_pre/entity_vector.model.bin

* fasttext
「@Hironsan さんの公開モデル」
  - https://qiita.com/Hironsan/items/513b9f93752ecee9e670
  - 学習データ: 日本語Wikipedia
  - 300次元
  
  置き場所: classifier/vec/fasttext_pre/fasttext.vec, classifier/vec/fasttext_pre/fasttext-neologd.vec

* Doc2vec
「Yuki Okuda さんの公開モデル」
  - https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/
  - 学習データ: 日本語Wikipedia
  - 300次元

  置き場所: classifier/vec/jawiki.doc2vec.dbow300d, classifier/vec/jawiki.doc2vec.dmpv300d

## 前処理, 特徴量作成等

### data/

* preprocess.py : 前処理
  出力ファイル : corpus-pre.tsv

* mecab_wakati.py : 分かち書き
  出力ファイル : corpus-wakati-mecab.tsv

* juman_wakati.py : 分かち書き
  出力ファイル : corpus-wakati-juman.tsv

JUMANの方が精度が良いので以降の分析ではこちらを使っている.

### vec/

* create_label.py : 感情ラベルの行列を作成
  出力ファイル : y_full.npy
* bow.py : 単語文章行列, n-gram行列の特徴量をs悪性
  出力ファイル : bow_train_x.npz, n-gram_x.npz
* bow-tf-idf.py : tf-idfの行列, n-gram+td-idfの行列の特徴量を作成
  出力ファイル : tf-idf_x.npz, n-gram-tf-idf_x.npz
* get_embedding_matrix.py : word2vec, fasttext の average, max, hier 特徴量を作成
  出力ファイル : [name]_aver.npy, [name]_max.npy, [name]_hier.npy
* get_doc2vec_matrix.py : doc2vec(PV-DM, PV-DBOW)の特徴量を作成
  出力ファイル : doc2vec-dbow.npy, doc2vec-dmpv.npy
* get_scdv.py : scdvの特徴量を作成
  出力ファイル : fasttext_scdv.npy
* sdv.py : sdv(極性辞書を使う方法)の特徴量を作成
  出力ファイル : sdv.npy

## Twitter Corpus 
* 評価方法: stratified 6 - fold
* 評価指標 : mean-F1
### (a). 単語カウント, 単語分散表現
| | Naive Beys(M / G) | Logistic Regression | LightGBM | MLP |
---- | --- | --- | --- | --- | --- |
Bow | 0.487159(M) | 0.46375 | 0.492416 | 0.4592
TF-IDF | 0.4540449(M) | 0.42702| 0.487880 | 0.45828
n-gram(n = 1,2,3) | 0.478082(M) | 0.47004| 0.492311 |  -
n-gram + TF-IDF | 0.473135(M) | 0.42517| 0.485405 | - 
word2vec&mean | 0.3363(G) | 0.444043 |0.437129 | 0.4535
word2vec&max | 0.32119(G) | 0.398972 | 0.429193| 0.4062
word2vec&mean+max | 0.34615(G) | 0.438370 | 0.444247 | 0.4510
word2vec&hier [Dinghan Shen et. 2018] | 0.316978(G) | 0.394948 | 0.411859 | 0.40671
fasttext&mean | 0.38369(G) | 0.466116 |  0.464673 | 0.49098
fasttext&max | 0.34234(G) | 0.440742 | 0.475811 | 0.45127
fasttext&mean+max | 0.382048(G)	 | 0.476638 | 0.477979 | 0.49097
fasttext&hier [Dinghan Shen et. 2018]| 0.33296(G) | 0.440843 | 0.443936 | 0.4556
SDV | | 0.47653 | 

* Naive Beys 
    * M : 多項分布を仮定
    * G : 正規分布を仮定.

### (b). 文章分散表現(直接求める)

| | Naive Beys(G) | Logistic Regression | LightGBM | MLP |
---- | --- | --- | --- | --- | --- |
Doc2Vec(PV-DBOW) | 0.33048 | 0.4617 | 0.45683 | 0.4680
Doc2vec(PV-DM)  | | 0.4063 | 0.4007 | 0.4050
Doc2vec(concat) | | 0.4688 |  | 0.4630

* BERTは本来fine-tuningして用いるものだが, 文章の先頭につけるタグ[CLS]が文章に対応する分散表現を獲得しているとみなして用いた.

### (c). Deep Model

| | Score |
---- | --- |
| LSTM(+fasttext) | 0.50552 |
| Bi-LSTM(+fasttext) | 0.51336 |
| GRU(+fasttext) | 0.50975 |


LSTM, Bi-LSTM, GRU のEmbedding層にはfasttextで得た分散表現を使用.学習を行わないようにした.
