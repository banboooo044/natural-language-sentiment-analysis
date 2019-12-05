from typing import Dict, Any, List, Optional

import itertools
import numpy as np
import sklearn
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, TfidfModel
from sklearn.mixture import GaussianMixture
import sklearn.base
from tqdm import tqdm


class SCDVVectorizer(sklearn.base.BaseEstimator):
    """ This is a model which is described in "SCDV : Sparse Composite Document Vectors using soft clustering over distributional representations"
    See https://arxiv.org/pdf/1612.06778.pdf for details
    
    """
    def __init__(self, embedding_size: int, sparsity_percentage: float,
                    gaussian_mixture_parameters: Dict[Any, Any], dictionary_filter_parameters: Dict[Any, Any] =  {}, 
                    word2vec_parameters: Dict[Any, Any] = {}, embedding_array: Optional[str] = None ) -> None:
        """
        :param documents: documents for training.
        :param embedding_size: word embedding size.
        :param cluster_size:  word cluster size.
        :param sparsity_percentage: sparsity percentage. This must be in [0, 1].
        :param word2vec_parameters: parameters to build `gensim.models.Word2Vec`. Please see `gensim.models.Word2Vec.__init__` for details.
        :param gaussian_mixture_parameters: parameters to build `sklearn.mixture.GaussianMixture`. Please see `sklearn.mixture.GaussianMixture.__init__` for details.
        :param dictionary_filter_parameters: parameters for `gensim.corpora.Dictionary.filter_extremes`. Please see `gensim.corpora.Dictionary.filter_extremes` for details.
        """

        self.embedding_size = embedding_size
        self.cluster_size = gaussian_mixture_parameters["n_components"]
        self.sparsity_percentage = sparsity_percentage
        self.word2vec_parameters = word2vec_parameters
        self.embedding_array = embedding_array
        gaussian_mixture_parameters.pop("n_components", None)
        self.gaussian_mixture_parameters = gaussian_mixture_parameters
        self.dictionary_filter_parameters = dictionary_filter_parameters

    def fit(self, documents: List[List[str]]):
        self._dictionary = self._build_dictionary(documents, self.dictionary_filter_parameters)
        vocabulary_size = len(self._dictionary.token2id)
        if self.embedding_array is None:
            self._word_embeddings = self._build_word_embeddings(documents, self._dictionary, self.embedding_size, self.word2vec_parameters)
        else:
            self._word_embeddings = np.load(self.embedding_array, allow_pickle=True)
    
        assert self._word_embeddings.shape == (vocabulary_size, self.embedding_size)

        self._word_cluster_probabilities = self._build_word_cluster_probabilities(self._word_embeddings, self.cluster_size, self.gaussian_mixture_parameters)
        assert self._word_cluster_probabilities.shape == (vocabulary_size, self.cluster_size)

        self._idf = self._build_idf(documents, self._dictionary)
        assert self._idf.shape == (vocabulary_size, )

        return self

    def transforms(self, documents: List[List[str]], l2_normalize: bool = True):
        vocabulary_size = len(self._dictionary.token2id)
        word_cluster_vectors = self._build_word_cluster_vectors(self._word_embeddings, self._word_cluster_probabilities)
        assert word_cluster_vectors.shape == (vocabulary_size, self.cluster_size, self.embedding_size)

        word_topic_vectors = self._build_word_topic_vectors(self._idf, word_cluster_vectors)
        assert word_topic_vectors.shape == (vocabulary_size, (self.cluster_size * self.embedding_size))

        document_vectors = self._build_document_vectors(word_topic_vectors, self._dictionary, documents)
        
        assert document_vectors.shape == (len(documents), self.cluster_size * self.embedding_size)

        self._sparse_threshold = self._build_sparsity_threshold(document_vectors, self.sparsity_percentage)

        return self._build_scdv_vectors(document_vectors, self._sparse_threshold, l2_normalize)

    def fit_transform(self, documents: List[List[str]], l2_normalize: bool = True):
        self.fit(documents)
        return self.transforms(documents, l2_normalize)

    def _build_dictionary(self, documents: List[List[str]], filter_parameters: Dict[Any, Any]) -> Dictionary:
        d = Dictionary(documents)
        d.filter_extremes(**filter_parameters)
        return d

    def _build_word_embeddings(self, documents: List[List[str]], dictionary: Dictionary, embedding_size: int,
                                word2vec_parameters: Dict[Any, Any]) -> np.ndarray:
        print("build word embeddings")
        w2v = Word2Vec(documents, size=embedding_size, **word2vec_parameters)
        embeddings = np.zeros((len(dictionary.token2id), w2v.vector_size))
        for token, idx in dictionary.token2id.items():
            embeddings[idx] = w2v.wv[token]
        return embeddings

    def _build_word_cluster_probabilities(self, word_embeddings: np.ndarray, cluster_size: int,
                                            gaussian_mixture_parameters: Dict[Any, Any]) -> np.ndarray:
        print("build word cluster probabilities")
        gm = GaussianMixture(n_components=cluster_size, **gaussian_mixture_parameters)
        gm.fit(word_embeddings)
        return gm.predict_proba(word_embeddings)

    def _build_idf(self, documents: List[List[str]], dictionary: Dictionary) -> np.ndarray:
        print("build tf-idf")
        corpus = [ dictionary.doc2bow(doc) for doc in documents]
        model = TfidfModel(corpus=corpus, dictionary=dictionary)
        idf = np.zeros(len(dictionary.token2id))
        for idx, value in tqdm(model.idfs.items()):
            idf[idx] = value
        return idf

    def _build_word_cluster_vectors(self, word_embeddings: np.ndarray, word_cluster_probabilities: np.ndarray) -> np.ndarray:
        print("build word cluster vectors")
        vocabulary_size, embedding_size = word_embeddings.shape
        cluster_size = word_cluster_probabilities.shape[1]
        assert vocabulary_size == word_cluster_probabilities.shape[0]

        wcv = np.zeros((vocabulary_size, cluster_size, embedding_size))
        wcp = word_cluster_probabilities
        for v, c in tqdm(itertools.product(range(vocabulary_size), range(cluster_size))):
            wcv[v][c] = wcp[v][c] * word_embeddings[v]
        return wcv

    def _build_word_topic_vectors(self, idf: np.ndarray, word_cluster_vectors: np.ndarray) -> np.ndarray:
        print("build word topic vectors")
        vocabulary_size, cluster_size, embedding_size = word_cluster_vectors.shape
        assert vocabulary_size == idf.shape[0]

        wtv = np.zeros((vocabulary_size, cluster_size * embedding_size))
        for v in tqdm(range(vocabulary_size)):
            wtv[v] = idf[v] * word_cluster_vectors[v].flatten()
        return wtv

    def _build_document_vectors(self, word_topic_vectors: np.ndarray, dictionary: Dictionary,
                                documents: List[List[str]]) -> np.ndarray:
        print("build document vectors")
        _, D = word_topic_vectors.shape
        ret = np.empty((0, D))
        for d in tqdm(documents):
            vec = [ word_topic_vectors[idx] * count for idx, count in dictionary.doc2bow(d)]
            if len(vec) > 0:
                ret = np.vstack((ret, np.sum(vec, axis=0)))
            else:
                ret = np.vstack((ret, np.zeros((1, D))))
        return ret

    def _build_sparsity_threshold(self, document_vectors: np.ndarray, sparsity_percentage) -> float:
        print("build sparsity threshold")
        def _abs_average_max(m: np.ndarray) -> float:
            return np.abs(np.average(np.max(m, axis=1)))

        t = 0.5 * (_abs_average_max(document_vectors) + _abs_average_max(-document_vectors))
        return sparsity_percentage * t

    def _build_scdv_vectors(self, document_vectors: np.ndarray, sparsity_threshold: float, l2_normalize: bool) -> np.ndarray:
        print("build scdv vectors")
        close_to_zero = np.abs(document_vectors) < sparsity_threshold
        document_vectors[close_to_zero] = 0.0
        if not l2_normalize:
            return document_vectors
        return sklearn.preprocessing.normalize(document_vectors, axis=1, norm='l2')