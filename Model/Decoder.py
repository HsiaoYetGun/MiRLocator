import dynet as dy

class LSTMDecoder(object):
	def __init__(self, model, vocab, options):
		self.model = model.add_subcollection('decoder')
		self.layers = options.num_layers
		self.input_dims = options.emb_dim
		self.context_dims = options.enc_output_dim
		self.hidden_dims = options.hidden_dim
		self.word_dims = options.emb_dim
		self.lstmbuilder = dy.VanillaLSTMBuilder(self.layers, self.input_dims + self.context_dims, self.hidden_dims, self.model)
		self.dropout, self.word_dropout = options.dropout_rate, options.word_dropout_rate
		
		self.Wp_p = self.model.add_parameters((self.input_dims, self.context_dims))
		self.bp_p = self.model.add_parameters((self.input_dims,))

		self.Wo_p = self.model.add_parameters((self.input_dims, self.input_dims + self.context_dims + self.hidden_dims))
		self.bo_p = self.model.add_parameters((self.input_dims,))

		self.E_p = self.model.add_parameters((vocab.size(), self.input_dims))
		self.b_p = self.model.add_parameters((vocab.size(),), init=dy.ConstInitializer(0))


	def init_params(self, src_encH, batch_size, train):
		if train:
			self.lstmbuilder.set_dropout(self.dropout)
		else:
			self.lstmbuilder.disable_dropout()
		self.Wp = self.Wp_p.expr()
		self.bp = self.bp_p.expr()
		self.Wo = self.Wo_p.expr()
		self.bo = self.bo_p.expr()
		self.E = self.E_p.expr()
		self.b = self.b_p.expr(False)
		self.batch_size = batch_size
		last_enc = dy.pick(src_encH, index=src_encH.dim()[0][-1] - 1, dim=1)
		init_state = dy.affine_transform([self.bp, self.Wp, last_enc])
		init_state = [init_state, dy.zeros((self.hidden_dims,), batch_size=batch_size)]
		self.dec_state = self.lstmbuilder.initial_state(init_state)

		if train:
			self.lstmbuilder.set_dropout_masks(batch_size)


	def next(self, word_idx, context, train, cur_state=None):
		embs = dy.pick_batch(self.E, word_idx)
		if train:
			embs = dy.dropout_dim(embs, 0, self.word_dropout)
		x = dy.concatenate([embs, context])
		if cur_state is None:
			self.dec_state = self.dec_state.add_input(x)
			next_state = self.dec_state
		else:
			next_state = cur_state.add_input(x)
		hidden = next_state.output()
		return hidden, embs, next_state


	def score(self, hidden, context, embs, train):
		output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([hidden, context, embs])])
		if train:
			output = dy.dropout(output, self.dropout)
		return dy.affine_transform([self.b, self.E, output])

