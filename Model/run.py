import Options as Opts
import numpy as np
from Corpus import BatchLoader, corpus_filter 
import evaluation, helpers, sys



def train(options):
	log = helpers.Logger(options.verbose)
	timer = helpers.Timer()
	# Load data =========================================================
	log.info('Reading corpora')
	# Read vocabs
	vocab = helpers.get_dictionaries(options)
	src_dic, trg_dic = vocab['src'], vocab['trg']
	# Read training
	train_src_data = src_dic.read_corpus(options.train_src)
	train_trg_data = trg_dic.read_corpus(options.train_dst)

	max_src_len, max_trg_len = options.max_src_len, options.max_trg_len
	if max_src_len > 0 or max_trg_len > 0:
		train_src_data, train_trg_data = corpus_filter(train_src_data, train_trg_data, max_src_len, max_trg_len)
		assert len(train_src_data) == len(train_trg_data), 'Size of source corpus and the target corpus must be the same!!'
	# Read validation
	valid_src_data = src_dic.read_corpus(options.valid_src)
	valid_trg_data = trg_dic.read_corpus(options.valid_dst)
	# Validation output
	if not options.valid_out:
		options.valid_out = helpers.exp_filename(options, 'valid.out')
	# Get target language model
	lang_model = helpers.get_language_model(options, train_trg_data, trg_dic.size())
	# Create model ======================================================
	log.info('Creating model')
	s2s = helpers.build_model(options, vocab, lang_model)

	# Trainer ==========================================================
	trainer = helpers.get_trainer(options, s2s)
	log.info('Using ' + options.trainer + ' optimizer')
	# Print configuration ===============================================
	if options.verbose:
		Opts.print_config(options, src_dict_size=src_dic.size(), trg_dict_size=trg_dic.size())
	# Creat batch loaders ===============================================
	log.info('Creating batch loaders')
	trainbatchloader = BatchLoader(train_src_data, train_trg_data, options.batch_size)
	devbatchloader = BatchLoader(valid_src_data, valid_trg_data, options.dev_batch_size)
	# Start training ====================================================
	log.info('starting training')
	timer.restart()
	train_loss = 0.
	processed = 0
	best_bleu = -1
	bleu = -1
	deadline = 0
	i = 0
	for epoch in xrange(options.num_epochs):
		for x, y in trainbatchloader:
			s2s.set_train_flag()
			processed += sum(map(len, y))
			bsize = len(y)
			# Compute loss
			loss = s2s.calculate_loss(x, y)
			# Backward pass and parameter update
			train_loss += loss.scalar_value() * bsize

			loss.backward()
			trainer.update()
		   
			if (i + 1) % options.check_train_error_every == 0:
				# Check average training error from time to time
				logloss = train_loss / processed
				ppl = np.exp(logloss)
				trainer.status()
				log.info(" Training_loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
						 (logloss, ppl, timer.tick(), processed))
				train_loss = 0
				processed = 0
			
			if (i + 1) % options.check_valid_error_every == 0:
				# Check generalization error on the validation set from time to time
				s2s.set_test_flag()
				dev_loss = 0
				dev_processed = 0
				timer.restart()
				for x, y in devbatchloader:
					dev_processed += sum(map(len, y))
					bsize = len(y)
					loss = s2s.calculate_loss(x, y)
					dev_loss += loss.scalar_value() * bsize
				dev_logloss = dev_loss / dev_processed
				dev_ppl = np.exp(dev_logloss)
				log.info("[epoch %d] Dev loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
						 (epoch, dev_logloss, dev_ppl, timer.tick(), dev_processed))

			if (i + 1) % options.valid_bleu_every == 0:
				# Check BLEU score on the validation set from time to time
				s2s.set_test_flag()
				log.info('Start translating validation set, buckle up!')
				timer.restart()
				with open(options.valid_out, 'w+') as fp:
					for x in valid_src_data:
						y_hat = s2s.translate(x, beam_size=options.beam_size)
						translation = [trg_dic.get_word(w) for w in y_hat[1: -1]]
						fp.write(' '.join(translation))
						fp.write('\n')
				bleu, details = evaluation.bleu_score(options.valid_dst, options.valid_out)
				log.info('Finished translating validation set %.2f elapsed.' % timer.tick())
				log.info(details)
				# Early stopping : save the latest best model
				if bleu > best_bleu:
					best_bleu = bleu
					log.info('Best BLEU score up to date, saving model to %s' % options.model)
					s2s.save(options.model)
					deadline = 0
				else:
					deadline += 1
				if options.patience > 0 and deadline > options.patience:
					log.info('No improvement since %d epochs, early stopping '
							 'with best validation BLEU score: %.3f' % (deadline, best_bleu))
					sys.exit()
		#	i += 1
		# trainer.update()

	#if bleu > best_bleu:
	#	s2s.save(options.model)
	s2s.save(options.model)

def test(options):
	log = helpers.Logger(options.verbose)
	timer = helpers.Timer()
	# Load data =========================================================
	log.info('Reading corpora')
	# Read vocabs
	vocab = helpers.get_dictionaries(options, test=True)
	src_dic, trg_dic = vocab['src'], vocab['trg']
	# Read test
	tests_data = src_dic.read_corpus(options.test_src)
	# Test output
	if not options.test_out:
		options.test_out = helpers.exp_filename(options, 'test.out')
	# Get target language model
	lang_model = helpers.get_language_model(options, None, trg_dic.size(), test=True)
	# Create model ======================================================
	log.info('Creating model')
	s2s = helpers.build_model(options, vocab, lang_model, test=True)
	# Print configuration ===============================================
	if options.verbose:
		Opts.print_config(options, src_dict_size=src_dic.size(), trg_dict_size=trg_dic.size())
	# Start testing =====================================================
	log.info('Start running on test set, buckle up!')
	timer.restart()
	translations = []
	s2s.set_test_flag()
	for i, x in enumerate(tests_data):
		y = s2s.translate(x, beam_size=options.beam_size)
		translations.append(' '.join([trg_dic.get_word(w) for w in y[1:-1]]))
	translations = np.asarray(translations, dtype=str)
	np.savetxt(options.test_out, translations, fmt='%s')
	if options.test_dst is not None:
		BLEU, details = evaluation.bleu_score(options.test_dst, options.test_out)
		log.info(details)
	log.info('Finished running on test set %.2f elapsed.' % timer.tick())


if __name__ == '__main__':
	# Retrieve options ==================================================
	options = Opts.get_options()
	if options.train:
		train(options)
	elif options.test:
		test(options)

