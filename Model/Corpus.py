import pickle
import numpy as np
from collections import defaultdict


class Dictionary(object):
	def __init__(self, dic):
		self.__w2i = dic
		self.__i2w = {v: k for k, v in dic.items()}


	def save(self, fn):
		with open(fn, 'w') as fp:
			pickle.dump(dict(self.__w2i), fp)
			pickle.dump(self.unk, fp)
			pickle.dump(self.sos, fp)
			pickle.dump(self.eos, fp)


	@classmethod
	def load(cls, fn):
		with open(fn, 'r') as fp:
			saved_dic = pickle.load(fp)
			unk = pickle.load(fp)
			sos = pickle.load(fp)
			eos = pickle.load(fp)
		vocab = cls(defaultdict(lambda: 0, saved_dic))
		vocab.unk, vocab.sos, vocab.eos = unk, sos, eos
		return vocab


	@classmethod
	def build(cls, fn, max_size=20000, min_freq=1, unk='UNK', sos='SOS', eos='EOS'):
		dic = defaultdict(lambda: 0)
		freqs = defaultdict(lambda: 0)
		dic[unk], dic[sos], dic[eos] = 0, 1, 2

		with open(fn, 'r') as rfp:
			for line in rfp:
				sent = line.strip().split()
				for word in sent:
					freqs[word] += 1

		sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
		for i in xrange(min(max_size, len(sorted_words))):
			word, freq = sorted_words[i]
			if freq <= min_freq:
				continue
			dic[word] = len(dic)

		vocab = cls(dic)
		vocab.unk, vocab.sos, vocab.eos = unk, sos, eos
		return vocab


	def read_corpus(self, fn):
		sentences = []
		sos_id = self.__w2i[self.sos]
		eos_id = self.__w2i[self.eos]
		unk_id = self.__w2i[self.unk]
		dic = self.__w2i

		with open(fn, 'r') as rfp:
			for line in rfp:
				line = line.strip().split()
				sent = [sos_id]
				for word in line:
					if word not in dic:
						sent.append(unk_id)
					else:
						sent.append(dic[word])
				sent.append(eos_id)
				sentences.append(sent)
		return sentences


	def get_index(self, word):
		return self.__w2i[word]


	def get_word(self, index):
		return self.__i2w[index]


	def size(self):
		return len(self)


	def __len__(self):
		return len(self.__w2i)


	def iteritems(self):
		return self.__w2i.iteritems()


	def eos_id(self):
		return self.__w2i[self.eos]


	def sos_id(self):
		return self.__w2i[self.sos]


	def unk_id(self):
		return self.__w2i[self.unk]


def __filt(corpus, key, max_len):
	filted = []
	for sent in corpus:
		if len(sent[key]) < max_len:
			filted.append(sent)
	return filted


def corpus_filter(src_corpus, trg_corpus, src_max_len=-1, trg_max_len=-1):
	corpus = zip(src_corpus, trg_corpus)
	if src_max_len > 0:
		corpus = __filt(corpus, 0, src_max_len)
	if trg_max_len > 0:
		corpus = __filt(corpus, 1, trg_max_len)
	return zip(*corpus)


class BatchLoader(object):
	def __init__(self, src_corpus, trg_corpus, batch_size):
		self.batches = []
		self.batch_size = batch_size

		buckets = defaultdict(list)
		for src, trg in zip(src_corpus, trg_corpus):
			buckets[len(src)].append((src, trg))

		for src_len, bucket in buckets.items():
			np.random.shuffle(bucket)
			num_bathes = int(np.ceil((len(bucket) + 0.0) / self.batch_size))
			for i in xrange(num_bathes):
				cur_batch_size = self.batch_size if i < (num_bathes - 1) else (len(bucket) - self.batch_size * i)
				self.batches.append(([bucket[i * self.batch_size + j][0] for j in xrange(cur_batch_size)], 
									[bucket[i * self.batch_size + j][1] for j in xrange(cur_batch_size)]))

		self.__size = len(self.batches)
		self.reseed()


	def reseed(self):
		print('Reseeding the dataset')
		self.__index = 0
		np.random.shuffle(self.batches)


	def next(self):
		if self.__index >= self.__size - 1:
			self.reseed()
			raise StopIteration()
		self.__index += 1
		return self.batches[self.__index]


	def __next__(self):
		return self.next()


	def __iter__(self):
		return self

