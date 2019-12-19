# natural-language-sentiment-analysis
文章からの感情解析

## 学習済みモデル
* Word2vec
「東北大学 乾・岡崎研究室の公開モデル」
  - http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
  - 学習データ: 日本語Wikipedia
  - 200次元

* fasttext
「@Hironsan さんの公開モデル」
  - https://qiita.com/Hironsan/items/513b9f93752ecee9e670
  - 学習データ: 日本語Wikipedia
  - 300次元

* Doc2vec
「Yuki Okuda さんの公開モデル」
  - https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/
  - 学習データ: 日本語Wikipedia
  - 300次元

* BERT
「京都大学 黒橋・河原研究室の公開モデル」
   - http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB
   - 学習データ: 日本語Wikipedia


## Twitter Corpus 
* 評価方法: stratified 6 - fold
* 評価指標 : mean-F1
### (a). 単語カウント, 単語分散表現
| | Naive Beys(M / G) | Logistic Regression | LightGBM | MLP
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

| | Naive Beys(G) | Logistic Regression | LightGBM | MLP
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
