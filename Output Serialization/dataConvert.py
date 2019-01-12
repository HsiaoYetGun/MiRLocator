import pandas as pd

def reorder(arg, ord):
    print(ord)
    df = pd.read_csv(arg.inputFile).iloc[:, 1 :]
    diff = df.shape[1] - len(ord)
    df2 = df.iloc[:, : diff]
    cnt = 0
    for loc in ord:
        l = df[loc]
        df2.insert(cnt + diff, loc, l)
        cnt += 1
    convert(df2, arg, ord)

def convert(df, arg, locList):
    ordS = ord('a')
    locRep = [chr(ordS + i) for i in range(len(locList) * 2)]
    df2 = pd.DataFrame()
    l = [[] for _ in range(df.shape[1])]
    cnt = 0
    c = 0
    length = len(df)
    seqList = []
    for location in locList:
        temp = df.loc[:, location].copy()
        temp[temp == 1] = locRep[c]
        c += 1
        temp[temp == 0] = locRep[c]
        c += 1
        l[cnt] = temp
        cnt += 1
    for i in range(length):
        seqOri = df.iloc[i, :]['seq']
        seqlen = len(seqOri)
        diffL = seqlen - arg.mer
        seq = ''
        for j in range(diffL):
            seq += seqOri[j : j + arg.mer] + ' '
        seqList.append(seq)
    name = df.loc[:, 'name']
    df2.insert(0, 'name', name)
    df2.insert(1, 'seq', seqList)
    for i in range(6):
        df2.insert(i + 2, locList[i], l[i])
    df2.to_csv(arg.convertFile)