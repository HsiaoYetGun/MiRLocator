import numpy as np


def read_data(fn):
	raw_data = np.loadtxt(fn, dtype=np.str, delimiter=',')
	pro_data = raw_data[1:, 2:]
	splitedData = [[] for _ in xrange(10)]
	np.random.shuffle(pro_data)
	for i in xrange(10):
		splitedData[i] = pro_data[i::10]
	for i in xrange(10):
		with open('%d_src.train' % i, 'w') as sfp, open('%d_trg.train' % i, 'w') as tfp:
			for j in xrange(10):
				if j != i:
					for sent in splitedData[j]:
						sfp.write('%s\n' % sent[0].replace('', ' '))
						tfp.write('%s\n' % ' '.join(sent[1:]))
		with open('%d_src.test' % i, 'w') as sfp, open('%d_trg.test' % i, 'w') as tfp:
			for sent in splitedData[i]:
				sfp.write('%s\n' % sent[0].replace('', ' '))
				tfp.write('%s\n' % ' '.join(sent[1:]))

read_data('data.csv')

