import config
import sortingAlgorithm
import dataConvert
import splitDataset

def preprocessing():
    conf = config.Preprocessing()
    conf.printInfo()

    sortingAlgorithm.labelStat(conf.arg)
    sortRes = None
    if conf.arg.method == 'ent':
        sortRes = sortingAlgorithm.entropySort(conf.arg)
    elif conf.arg.method == 'ent2':
        sortDict = sortingAlgorithm.entropySort2(conf.arg)[0]
        sortRes = sortDict.split(',')
    elif conf.arg.method == 'pearson':
        sortRes = sortingAlgorithm.pearsonSort(conf.arg)
    elif conf.arg.method == 'occ':
        sortRes = sortingAlgorithm.occSort(conf.arg)

    dataConvert.reorder(conf.arg, sortRes)

    splitDataset.read_data(conf.arg)

if __name__ == '__main__':
    preprocessing()