import dynet as dy
import numpy as np

from Attention import *
from Encoder import *
from Decoder import *


class Beam(object):
	def __init__(self, state, context, words, log_prob):
		self.state = state
		self.context = context
		self.words = words
		self.log_prob = log_prob


class Seq2SeqModel(object):
	def __init__(self, options, vocab, lang_model):
		self.model = dy.ParameterCollection()
		src_dic = vocab['src']
		trg_dic = vocab['trg']
		self.__enc = BiLSTMEncoder(self.model, src_dic, options)
		options.enc_output_dim = self.__enc.output_dim
		self.__dec = LSTMDecoder(self.model, trg_dic, options)
		self.__att = MLPAttention(self.model, options)
		self.__lm = lang_model
		self.__train_flag = False
		self.__src_eos = src_dic.eos_id()
		self.__trg_sos = trg_dic.sos_id()
		self.__trg_eos = trg_dic.eos_id()
		self.__max_len = options.max_len
		self.__ls, self.__ls_eps = options.label_smoothing > 0, options.label_smoothing
		self.__trg_vsize = trg_dic.size()

	def prepare_batch(self, batch, eos):
		batch_size = len(batch)
		self.batch_size = batch_size
		batch_len = max(len(s) for s in batch)
		x = np.zeros((batch_len, batch_size), dtype=int)
		masks = np.ones((batch_len, batch_size), dtype=float)
		x[:] = eos
		
		for i in xrange(batch_size):
			sent = batch[i][:]
			masks[len(sent):, i] = 0
			while len(sent) < batch_len:
				sent.append(eos)
			x[:, i] = sent
		return x, masks


	def encode(self, src):
		x, _ = self.prepare_batch(src, self.__src_eos)
		self.__enc.init_params(self.batch_size, self.__train_flag)
		return self.__enc.encode(x)


	def attend(self, encodings, trg_h):
		self.__att.init_params()
		return self.__att.attend(encodings, trg_h)


	def cross_entropy_loss(self, score, next_word, cur_word):
		if self.__ls:
			log_prob = dy.log_softmax(score)
			if self.__lm is None:
				loss = - dy.pick_batch(log_prob, next_word) * (1 - self.__ls_eps) - \
					dy.mean_elems(log_prob) * self.__ls_eps
			else:
				loss = - dy.pick_batch(log_prob, next_word) * (1 - self.__ls_eps) - \
					dy.dot_product(self.__lm.next_expr(cur_word), log_prob) * self.__ls_eps
		else:
			loss = dy.pickneglogsoftmax(score, next_word)
		return loss


	def decode_loss(self, encodings, trg):
		y, masksy = self.prepare_batch(trg, self.__trg_eos)
		slen, batch_size = y.shape
		self.__dec.init_params(encodings, batch_size, self.__train_flag)
		context = dy.zeros((self.__enc.output_dim, ), batch_size=batch_size)

		errs = []
		for cur_word, next_word, mask in zip(y, y[1:], masksy[1:]):
			hidden, embs, _ = self.__dec.next(cur_word, context, self.__train_flag)
			context, _ = self.attend(encodings, hidden)
			score = self.__dec.score(hidden, context, embs, self.__train_flag)
			masksy_embs = dy.inputTensor(mask, batched=True)

			loss = self.cross_entropy_loss(score, next_word, cur_word)
			loss = dy.cmult(loss, masksy_embs)
			errs.append(loss)

		error = dy.mean_batches(dy.esum(errs))
		return error


	def calculate_loss(self, src, trg):
		dy.renew_cg()
		encodings = self.encode(src)
		error = self.decode_loss(encodings, trg)
		return error


	def translate(self, x, beam_size=1):
		dy.renew_cg()
		encodings = self.encode([x])
		return self.beam_decode(encodings, input_len=len(x), beam_size=beam_size)


	def beam_decode(self, encodings, input_len=10, beam_size=1):
		batch_size = 1
		self.__dec.init_params(encodings, batch_size, self.__train_flag)
		context = dy.zeros((self.__enc.output_dim, ))
		beams = [Beam(self.__dec.dec_state, context, [self.__trg_sos], 0.0)]

		for i in xrange(int(min(self.__max_len, input_len * 1.5))):
			new_beams = []
			p_list = []
			for b in beams:
				if b.words[-1] == self.__trg_eos:
					p_list.append(dy.ones((self.__trg_vsize, )))
					continue
				hidden, embs, b.state = self.__dec.next([b.words[-1]], b.context, self.__train_flag, b.state)
				b.context, _ = self.attend(encodings, hidden)
				score = self.__dec.score(hidden, b.context, embs, self.__train_flag)
				p_list.append(dy.softmax(score))
			p_list = dy.concatenate_to_batch(p_list).npvalue().T.reshape(-1, self.__trg_vsize)
			for p, b in zip(p_list, beams):
				p = p.flatten() / p.sum()
				kbest = np.argsort(p)
				if b.words[-1] == self.__trg_eos:
					new_beams.append(Beam(b.state, b.context, b.words, b.log_prob))
				else:
					for next_word in kbest[-beam_size:]:
						new_beams.append(Beam(b.state, b.context, b.words + [next_word], b.log_prob + np.log(p[next_word])))
			beams = sorted(new_beams, key=lambda b: b.log_prob)[-beam_size:]
			if beams[-1].words[-1] == self.__trg_eos:
				break
		return beams[-1].words


	def save(self, fn):
		self.model.save(fn)


	def load(self, fn):
		self.model.populate(fn)


	def set_train_flag(self):
		self.__train_flag = True


	def set_test_flag(self):
		self.__train_flag = False



