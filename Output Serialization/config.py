import argparse
import yaml
import sys

class Preprocessing():
    def __init__(self):
        self.__parser = argparse.ArgumentParser()
        self.arg = None

        self.__addArguments()
        self.__readConfig()

    def __addArguments(self):
        # Mode
        self.__parser.add_argument('--method',
                                   help = 'Sorting Algorithm : [occ, pearson, ent, ent2]',
                                   choices = ['occ', 'pearson', 'ent', 'ent2'],
                                   type = str)
        self.__parser.add_argument('--order',
                                   help = 'Sort Order : [asc, desc]',
                                   choices = ['asc', 'desc'],
                                   type = str)
        self.__parser.add_argument('--mer',
                                   help = 'Mer Length',
                                   type = int)

        # IO
        self.__parser.add_argument('--inputFile',
                                   '-if',
                                   help = 'Input File',
                                   type = str)
        self.__parser.add_argument('--outputDir',
                                   '-od',
                                   help = 'Output Directory',
                                   type = str)
        self.__parser.add_argument('--convertFile',
                                   '-cf',
                                   help = 'Convert File',
                                   type = str)
        self.__parser.add_argument('--probFile',
                                   '-pf',
                                   help = 'Probability File',
                                   type = str)

        # Config File
        self.__parser.add_argument('--config',
                                   '-c',
                                   help = 'Config File',
                                   default = './config/preprocessing.yaml',
                                   type = str)

    def __readConfig(self):
        arg = self.__parser.parse_args()
        with open(arg.config) as conf:
            configDict = yaml.load(conf)
            for key, value in configDict.items():
                sys.argv.append('--' + key)
                sys.argv.append(str(value))
        self.arg = self.__parser.parse_args()

    def printInfo(self):
        argDict = vars(self.arg)
        print('-' * 20 + ' Config Information ' + '-' * 20)
        for key, value in argDict.items():
            print('%-12s : %s'% (key, value))
        print('-' * 60)