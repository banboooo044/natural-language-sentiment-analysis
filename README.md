# natural-language-sentiment-analysis

## 1. 学習済みモデルのダウンロード
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

## 2. セットアップ方法
github上には, メモリの関係上元データしかアップしていないので, 以下の手順でプログラムを実行することで復元できる.

1. 以下をコマンドラインで実行.
   ```
   git clone https://github.com/banboooo044/natural-language-sentiment-analysis.git
   ```
2. 学習済みモデルのダウンロードし, 指定の置き場所にファイルを置く.
3. 以下をコマンドラインで実行.前処理と分かち書きが行われたファイルが作成される.
  ```bash
    bash data/setup.sh
  ```
  
4. 以下でコマンドラインを実行.特徴量の行列ファイルが作成される.
  ```bash
    bash vec/setup.sh
  ```

## 3. 前処理, 特徴量作成のプログラム

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
* bow.py : 単語文章行列, n-gram行列の特徴量を作成

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

## 5. 分析プログラム

### src/ : 汎用的なプログラム(他のプログラムから呼び出して使う再利用性の高いもの)

* model.py : Modelクラスは学習, 予測, モデルの保存やロードを行う.Modelクラスを継承して, 分類アルゴリズムごとのクラスを作る.
  * model_NB : ナイーブベイズ(多項分布モデル)
  * model_GaussNB : ナイーブベイズ(ガウス分布モデル)
  * model_logistic.py : ロジスティック回帰
  * model_lgb.py : LightGBM
  * model_MLP ; Multilayer Perceptron
  * model_LSTM : LSTM
  * model_GRU : GRU
  
* runner.py : Runnerクラスはクロスバリデーションなども含めた学習, 評価, 予測を行うためのクラス.Modelクラスを継承しているプログラムを渡す.
* util.py : ファイルの入出力, ログの出力や表示, 計算結果の表示や出力を行うクラス

### code-analysis/ : コード分析用のプログラム

* run_[アルゴリズム名].py : Runnerクラスを用いて, 実際に各分類アルゴリズムで学習を行うプログラム.
* [アルゴリズム名]_gridCV.py : グリッドサーチでアルゴリズムのパラメータチューニングを行うプログラム.
* [アルゴリズム名]_tuning.py : hyperopt(ベイズ最適化を用いたパラメータ自動探索ツール)を用いてパラメータチューニングを行うプログラム

## 4. 実験(Twitter Corpus)

* 評価方法: stratified 6 - fold
* 評価指標 : mean-F1

### (a). 単語カウント, 単語分散表現
| | Naive Beys(M / G) | Logistic Regression | LightGBM | MLP |
|-| ----------------- | --------------------| -------- | --- |
|Bow| 0.487159(M) | 0.46375 | 0.492416 | 0.4592 |
|TF-IDF | 0.4540449(M) | 0.42702| 0.487880 | 0.45828 |
|n-gram(n = 1,2,3) | 0.478082(M) | 0.47004| 0.492311 | - |
|n-gram + TF-IDF | 0.473135(M) | 0.42517| 0.485405 | - |
|word2vec&mean | 0.3363(G) | 0.444043 |0.437129 | 0.4535 |
|word2vec&max | 0.32119(G) | 0.398972 | 0.429193| 0.4062 |
|word2vec&mean+max | 0.34615(G) | 0.438370 | 0.444247 | 0.4510 |
|word2vec&hier [Dinghan Shen et. 2018] | 0.316978(G) | 0.394948 | 0.411859 | 0.40671 |
|fasttext&mean | 0.38369(G) | 0.466116 |  0.464673 | 0.49098 |
|fasttext&max | 0.34234(G) | 0.440742 | 0.475811 | 0.45127 |
|fasttext&mean+max | 0.382048(G) | 0.476638 | 0.477979 | 0.49097 |
|fasttext&hier [Dinghan Shen et. 2018]| 0.33296(G) | 0.440843 | 0.443936 | 0.4556 |
|SDV | - | 0.47653 | 0.48 | 0.48 | 

* Naive Beys 
    * M : 多項分布を仮定
    * G : 正規分布を仮定.

### (b). 文章分散表現(直接求める)

| | Naive Beys(G) | Logistic Regression | LightGBM | MLP |
| ---- | --- | --- | --- | --- |
|Doc2Vec(PV-DBOW) | 0.33048 | 0.4617 | 0.45683 | 0.4680 |
|Doc2vec(PV-DM)   | | 0.4063 | 0.4007 | 0.4050 |
|Doc2vec(concat) | - | 0.4688 | 0.46 | 0.4630 |

### (c). Deep Model

| | Score |
|---- | --- |
| LSTM(+fasttext) | 0.50552 |
| Bi-LSTM(+fasttext) | 0.51336 |
| GRU(+fasttext) | 0.50975 |


LSTM, Bi-LSTM, GRU のEmbedding層にはfasttextで得た分散表現を使用.学習を行わないようにした.

