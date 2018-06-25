import csv
import numpy as np
from io import open
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

seed = 3535999445

def _atec(path):
    with open(path,'r',encoding='utf-8') as f:
        f = csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(list(f)):
            if i > 0:
                c1 = line[0]
                c2 = line[1]
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1]))
        return ct1, ct2, y

def atec(data_dir):
    comps1, comps2, ys = _atec(data_dir)
    trX1, trX2 = [], []
    trY = []
    for c1, c2, y in zip(comps1, comps2, ys):
        trX1.append(c1)
        trX2.append(c2)
        trY.append(y)

    trY = np.asarray(trY, dtype=np.int32)
    return (trX1, trX2, trY)
