from Corpus import *
from Seq2Seq import *
from LanguageModel import *
import dynet as dy
import time, sys

class Logger(object):
	def __init__(self, verbose=False):
		self.verbose = verbose

	
	def info(self, string):
		if self.verbose:
			print(string)
		sys.stdout.flush()


class Timer(object):
	def __init__(self, verbose=False):
		self.start = time.time()
	
	
	def restart(self):
		self.start = time.time()
	
	
	def elapsed(self):
		return time.time() - self.start

	
	def tick(self):
		elapsed = self.elapsed()
		self.restart()
		return elapsed


def exp_filename(options, name):
	return '%s/%s_%s' % (options.output_dir, options.exp_name, name)


def build_model(options, vocab, lang_model, test=False):
	s2s = Seq2SeqModel(options, vocab, lang_model)
	if test:
		if options.model is None:
			options.model = exp_filename(options, 'model')
		s2s.load(options.model)
	else:
		if options.model is not None:
			s2s.load(options.model)
		else:
			options.model = exp_filename(options, 'model')
	return s2s


def get_language_model(options, train_data, voc_size, test=False):
	if options.language_model is None or options.language_model == 'None':
		print('Do not use any language model.')
		return None
	if options.language_model == 'uniform':
		lang_model = UniformLanguageModel(voc_size)
	elif options.language_model == 'unigram':
		lang_model = UnigramLanguageModel(voc_size)
	elif options.language_model == 'bigram':
		lang_model = BigramLanguageModel(voc_size)
	else:
		print('Unknown language model %s, using unigram language model' % options.language_model)
		lang_model = UnigramLanguageModel(voc_size)

	if options.lm_file is not None or test:
		if options.lm_file is None:
			options.lm_file = exp_filename(options, 'lm')
		lang_model.load(options.lm_file)
	else:
		print('training language model...')
		lang_model.fit(train_data)
		options.lm_file = exp_filename(options, 'lm')
		lang_model.save(options.lm_file)
	return lang_model


def get_dictionaries(options, test=False):
	if options.dic_src:
		dic_src = Dictionary.load(options.dic_src)
	elif options.train_src or not test:
		dic_src = Dictionary.build(options.train_src, max_size=options.src_vocab_size, min_freq=options.min_freq)
		dic_src.save(exp_filename(options, 'src_dic'))
	else:
		dic_src = Dictionary.load(exp_filename(options, 'src_dic'))
	
	if options.dic_dst:
		dic_dst = Dictionary.load(options.dic_dst)
	elif options.train_dst or not test:
		dic_dst = Dictionary.build(options.train_dst, max_size=options.trg_vocab_size, min_freq=options.min_freq)
		dic_dst.save(exp_filename(options, 'trg_dic'))
	else:
		dic_dst = Dictionary.load(exp_filename(options, 'trg_dic'))
	return {'src': dic_src, 'trg': dic_dst}


def get_trainer(options, s2s):
	if options.trainer == 'sgd':
		trainer = dy.SimpleSGDTrainer(s2s.model, learning_rate=options.learning_rate)
	elif options.trainer == 'clr':
		trainer = dy.CyclicalSGDTrainer(s2s.model, e0_min=options.learning_rate / 10.0, e0_max=options.learning_rate, edecay=options.learning_rate_decay)
	elif options.trainer == 'momentum':
		trainer = dy.MomentumSGDTrainer(s2s.model, e0=options.learning_rate, edecay=options.learning_rate_decay)
	elif options.trainer == 'rmsprop':
		trainer = dy.RMSPropTrainer(s2s.model, e0=options.learning_rate, edecay=options.learning_rate_decay)
	elif options.trainer == 'adam':
		trainer = dy.AdamTrainer(s2s.model, options.learning_rate)
	else:
		print >> sys.stderr, 'Trainer name invalid or not provided, using SGD'
		trainer = dy.SimpleSGDTrainer(s2s.model, e0=options.learning_rate, edecay=options.learning_rate_decay)
	trainer.set_clip_threshold(options.gradient_clip)
	return trainer
