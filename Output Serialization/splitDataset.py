import numpy as np
import pandas as pd
import os
import re

def read_data(arg):
    raw_data = np.loadtxt(arg.convertFile, dtype=np.str, delimiter=',')
    nameList = [[] for _ in range(1)]
    pro_data = raw_data[1:, 1:]
    splitedData = [[] for _ in xrange(1)]
    np.random.shuffle(pro_data)
    for i in xrange(1):
        splitedData[i] = pro_data[i::1]
        v = str(splitedData[i][:, 0])
        v = v.split(' ')
        pattern = re.compile("'(hsa-.*)'")
        for value in v:
            tmp = pattern.findall(value)
            nameList[i].append(tmp[0])
        splitedData[i] = splitedData[i][:, 1:]
    i = 0
    for v in splitedData[0]:
        print(i, v)
        i += 1
    if not os.path.exists(arg.outputDir):
        os.makedirs(arg.outputDir)
    for i in xrange(1):
        '''
        with open('%s/%d_src.train' % (arg.outputDir, i), 'w') as sfp, open('%s/%d_trg.train' % (arg.outputDir, i), 'w') as tfp:
            for j in xrange(1):
                if j != i:
                    for sent in splitedData[j]:
                        sfp.write('%s\n' % sent[0])
                        tfp.write('%s\n' % ' '.join(sent[1:]))
        '''
        with open('%s/%d_src.test' % (arg.outputDir, i), 'w') as sfp, open('%s/%d_trg.test' % (arg.outputDir, i), 'w') as tfp:
            for sent in splitedData[i]:
                sfp.write('%s\n' % sent[0])
                tfp.write('%s\n' % ' '.join(sent[1:]))

    for i in xrange(1):
        ndf = pd.DataFrame()
        ndf.insert(0, 'name', nameList[i])
        ndf.to_csv('%s/%d_name.csv' % (arg.outputDir, i))