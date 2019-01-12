import dynet as dy

class BiLSTMEncoder(object):
	def __init__(self, model, vocab, options):
		self.model = model.add_subcollection('encoder')
		self.layers = options.num_layers
		self.input_dims = options.emb_dim
		self.hidden_dims = options.hidden_dim
		self.word_dims = options.emb_dim
		self.lstmbuilders = [dy.VanillaLSTMBuilder(self.layers, self.input_dims, self.hidden_dims, self.model), 
							dy.VanillaLSTMBuilder(self.layers, self.input_dims, self.hidden_dims, self.model)]
		self.dropout = options.dropout_rate
		self.WORD_LOOKUP = self.model.add_lookup_parameters((vocab.size(), self.word_dims))
		self.output_dim = self.hidden_dims * 2

		src_extrn_emb = None
		if options.src_extrn is not None:
			with open(options.src_extrn, 'r') as efp:
				wfp.readline()
				src_extrn_emb = {line.split(' ')[0]: float(emb) for emb in line.strip().split(' ')[1:] for line in efp}
			for word, idx in vocab.iteritems():
				if word in src_extrn_emb:
					self.WORD_LOOKUP.init_raw(idx, src_extrn_emb[word])


	def init_params(self, batch_size, train):
		if train:
			self.lstmbuilders[0].set_dropout(self.dropout)
			self.lstmbuilders[1].set_dropout(self.dropout)
		else:
			self.lstmbuilders[0].disable_dropout()
			self.lstmbuilders[1].disable_dropout()
		self.batch_size = batch_size
		self.init_states = [self.lstmbuilders[0].initial_state(), 
							self.lstmbuilders[1].initial_state()]
		if train:
			self.lstmbuilders[0].set_dropout_masks(self.batch_size)
			self.lstmbuilders[1].set_dropout_masks(self.batch_size)


	def encode(self, x):
		wembs = [dy.lookup_batch(self.WORD_LOOKUP, wi) for wi in x]
		encode_states = [dy.concatenate_cols(self.init_states[0].transduce(wembs)), 
						dy.concatenate_cols(self.init_states[1].transduce(wembs[::-1])[::-1])]
		return dy.concatenate(encode_states)

