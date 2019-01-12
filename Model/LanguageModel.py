import numpy as np
import dynet as dy
from scipy import sparse as sp
from collections import defaultdict
import pickle


class LanguageModel(object):
	def __init__(self, voc_size, grams, lm_type='lang'):
		self._voc_size = voc_size
		self._grams = grams
		self._lm_type = lm_type


	def next_expr(self, cur_word):
		return dy.inputTensor(self._next(cur_word), batched=(self._lm_type=='bigrams'))


	def save(self, fn):
		with open('%s%s' % (fn, self._lm_type), 'wb') as wfp:
			pickle.dump(self._grams, wfp)


	def load(self, fn):
		with open('%s%s' % (fn, self._lm_type)) as rfp:
			self._grams = pickle.load(rfp)


class UniformLanguageModel(LanguageModel):
	def __init__(self, voc_size, grams=None):
		super(UniformLanguageModel, self).__init__(voc_size, grams, 'uniform')


	def _next(self, cur_word):
		return self._grams


	def fit(self, corpus):
		self._grams = np.ones(self._voc_size) / self._voc_size


class UnigramLanguageModel(LanguageModel):
	def __init__(self, voc_size, eps=0, grams=None):
		super(UnigramLanguageModel, self).__init__(voc_size, grams, 'unigrams')
		self.__eps = eps


	def _next(self, cur_word):
		return self._grams


	def fit(self, corpus):
		self._grams = np.zeros(self._voc_size) + self.__eps
		for sent in corpus:
			for word in sent:
				self._grams[word] += 1
		self._grams /= self._grams.sum()



class BigramLanguageModel(LanguageModel):
	def __init__(self, voc_size, grams=None):
		super(BigramLanguageModel, self).__init__(voc_size, grams, 'bigrams')


	def _next(self, cur_word):
		return self._grams[cur_word].toarray().T


	# def next_expr(self, cur_word):
	# 	return dy.inputTensor(self._next(cur_word), batched=True)


	def fit(self, corpus):
		bigrams = defaultdict(lambda: defaultdict(lambda: 0.0))
		for sent in corpus:
			for word, next_word in zip(sent[:-1], sent[1:]):
				bigrams[word][next_word] += 1
		data, x, y = [], [], []
		for k, v in bigrams.items():
			s = sum(map(lambda x: x[1], v.items()))
			for w in v.keys():
				bigrams[k][w] /= s
				data.append(bigrams[k][w])
				x.append(k)
				y.append(w)
		voc_size = self._voc_size
		self._grams = sp.csr_matrix((data, (x, y)), shape=(voc_size, voc_size), dtype=float)

