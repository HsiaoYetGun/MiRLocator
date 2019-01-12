import pandas as pd
import numpy as np
from itertools import combinations, permutations
import json
import copy
from scipy import stats

locList = ['Exosome', 'Cytoplasm', 'Mitochondrion', 'Microvesicle', 'Circulating', 'Nucleus']

def labelStat(arg):
    df = pd.read_csv(arg.inputFile)
    N = df.shape[0]
    probList = []
    elemList = []
    with open(arg.probFile, 'w') as f:
        # prob
        tmp = combinations([0, 1, 2, 3, 4, 5], 1)
        posIndex = [list(v) for v in tmp]
        for index in posIndex:
            tdf = df[locList[index[0]]]
            probList.append(float(np.sum(tdf)) / N)
        elemList.extend(locList)

        # cond prob
        tmp = combinations([0, 1, 2, 3, 4, 5], 2)
        posIndex = [list(v) for v in tmp]
        for index in posIndex:
            a, b = index[0], index[1]
            tdf = df[[locList[a], locList[b]]]
            tData = np.asarray(tdf)
            tList = [list(v) for v in tData]
            cnt_ab = tList.count([1, 1])
            cnt_a = tList.count([1, 0]) + cnt_ab
            cnt_b = tList.count([0, 1]) + cnt_ab

            str_a = locList[a]
            str_b = locList[b]

            # b
            ## a | b
            if str(str_a + '|' + str_b) not in elemList:
                if cnt_b != 0:
                    probList.append(float(cnt_ab) / cnt_b)
                else:
                    probList.append(0)
                elemList.append(str_a + '|' + str_b)
            ## ~a | b
            if str('~' + str_a + '|' + str_b) not in elemList:
                if cnt_b != 0:
                    probList.append(float(cnt_b - cnt_ab) / cnt_b)
                else:
                    probList.append(0)
                elemList.append('~' + str_a + '|' + str_b)

            # a
            ## b | a
            if str(str_b + '|' + str_a) not in elemList:
                if cnt_a != 0:
                    probList.append(float(cnt_ab) / cnt_a)
                else:
                    probList.append(0)
                elemList.append(str_b + '|' + str_a)
            ## ~b | a
            if str('~' + str_b + '|' + str_a) not in elemList:
                if cnt_a != 0:
                    probList.append(float(cnt_a - cnt_ab) / cnt_a)
                else:
                    probList.append(0)
                elemList.append('~' + str_b + '|' + str_a)

            # ~b
            ## a | ~b
            if str(str_a + '|~' + str_b) not in elemList:
                probList.append(float(cnt_a - cnt_ab) / (df.shape[0] - cnt_b))
                elemList.append(str_a + '|~' + str_b)

            # ~a
            ## b | ~a
            if str(str_b + '|~' + str_a) not in elemList:
                probList.append(float(cnt_b - cnt_ab) / (df.shape[0] - cnt_a))
                elemList.append(str_b + '|~' + str_a)

        d = dict(zip(elemList, probList))
        json.dump(d, f)

def readProbAsDict(path):
    f = json.load(open(path, 'r'))
    probDict = {}
    condProbDict = dict(zip(locList, [{} for _ in range(len(locList))]))
    for loc in locList:
        condProbDict['~' + loc] = {}
    for key, value in f.items():
        key = str(key)
        if '|' not in key:
            probDict[key] = float(value)
            probDict['~' + key] = 1 - float(value)
        else:
            a, b = key.split('|')[:]
            condProbDict[b][a] = float(value)
    return probDict, condProbDict

def entropySort(arg):
    probDict, condProbDict = readProbAsDict(arg.probFile)
    resList = []
    first = True
    for location in locList:
        tmp = [location]
        if first == True:
            t = location
        tmpList = copy.copy(locList)
        tmpList.remove(location)
        d = condProbDict[location]
        while tmpList:
            entList = []
            for loc in tmpList:
                entropy = -d[loc] * np.log(d[loc]) - d['~' + loc] * np.log(d['~' + loc])
                entropy *= probDict[t]
                e2 = -probDict[loc] * np.log(probDict[loc]) - (1-probDict[loc]) * np.log(1-probDict[loc]) - entropy
                entList.append(e2)
            selectedLoc = tmpList[entList.index(max(entList))] if arg.order == 'desc' else tmpList[entList.index(min(entList))]
            tmp.append(selectedLoc)
            tmpList.remove(selectedLoc)
            t = selectedLoc

        resList.append(tmp)

    #for r in resList:
    #    print(r)
    #print('-' * 60)
    for v in resList:
        print(v)
    return resList[1]

def pearsonSort(arg):
    df = pd.read_csv(arg.inputFile)
    tmp = combinations(list(range(len(locList))), 2)
    index = [list(v) for v in tmp]
    resDict = {}
    for v in index:
        l1 = list(df[locList[v[0]]])
        l2 = list(df[locList[v[1]]])
        p, _ = stats.pearsonr(l1, l2)
        resDict[locList[v[0]] + '_' + locList[v[1]]] = np.abs(p)

    sortRes = sorted(resDict.items(), key = lambda d : d[1], reverse = (arg.order == 'desc'))
    used = [False] * len(locList)
    startLoc = sortRes[0][0].split('_')[0]
    resList = []
    resList.append(startLoc)
    used[locList.index(startLoc)] = True
    while False in used:
        for v in sortRes:
            loc = v[0].split('_')
            if startLoc in loc:
                ind = 1 - loc.index(startLoc)
                newLoc = loc[ind]
                if used[locList.index(newLoc)] == False:
                    startLoc = newLoc
                    resList.append(startLoc)
                    used[locList.index(startLoc)] = True
    return resList

def occSort(arg):
    df = pd.read_csv(arg.inputFile)
    srcData = np.asarray(df[locList])
    srcList = [list(v) for v in srcData]
    statDict = {}
    tmp = combinations(list(range(len(locList))), 2)
    posIndex = [list(v) for v in tmp]
    for index in posIndex:
        tag = [1 if v in index else 0 for v in range(6)]
        cnt = srcList.count(tag)
        if cnt != 0:
            statDict[str(tag)] = cnt

    sortRes = sorted(statDict.items(), key = lambda d : d[1], reverse = (arg.order == 'desc'))
    used = [False] * len(locList)
    resList = []
    for v in sortRes:
        v = v[0][1 : -1].split(',')
        v = [int(value) for value in v]
        ind1 = v.index(1)
        ind2 = v.index(1, ind1 + 1)
        if used[ind1] == False:
            used[ind1] = True
            resList.append(locList[ind1])
        if used[ind2] == False:
            used[ind2] = True
            resList.append(locList[ind2])
    return resList

def _binary_entropy(p):
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def _gain_eval(sequence, con_probs, probs):
    seq_scr = {}
    for seq in permutations(sequence, len(sequence)):
        gain = 0.
        for pre, cur in zip(seq[:-1], seq[1:]):
            p_p = probs[pre]
            p_c = probs[cur]
            npre = '~%s' % pre
            p_c_np = con_probs[npre][cur]
            p_c_p = con_probs[pre][cur]
            h_np = _binary_entropy(p_c_np)
            h_p = _binary_entropy(p_c_p)
            h_p_c = p_p * h_p + (1 - p_p) * h_np
            h_c = _binary_entropy(p_c)
            gain += (h_c - h_p_c)
        seq_scr[','.join(seq)] = gain
    return seq_scr

def entropySort2(arg, top = 50):
    probDict, condProbDict = readProbAsDict(arg.probFile)
    seq_scr = _gain_eval(locList, condProbDict, probDict)
    sorted_seq = sorted(seq_scr.items(), key=lambda x:x[1], reverse=True)
    res = None
    for e in sorted_seq[:top]:
        if e[0].startswith('Cytoplasm'):
            res = e
    return res