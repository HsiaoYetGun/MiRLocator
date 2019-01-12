import dynet as dy


class MLPAttention(object):
	def __init__(self, model, options):
		self.model = model.add_subcollection('attention')
		self.input_dims = options.enc_output_dim
		self.hidden_dims = options.hidden_dim
		self.atten_dims = options.att_dim

		self.Va_p = self.model.add_parameters((self.atten_dims))
		self.Wia_p = self.model.add_parameters((self.atten_dims, self.input_dims))
		self.Wha_p = self.model.add_parameters((self.atten_dims, self.hidden_dims))


	def init_params(self):
		self.Va = self.Va_p.expr()
		self.Wia = self.Wia_p.expr()
		self.Wha = self.Wha_p.expr()


	def attend(self, src_H, trg_h):
		hidden = dy.tanh(dy.colwise_add(self.Wia * src_H, self.Wha * trg_h))
		attn_scrs = dy.transpose(hidden) * self.Va
		weights = dy.softmax(attn_scrs)
		context = src_H * weights
		return context, weights