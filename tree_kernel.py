from __future__ import print_function
import os
import numpy as np
DATA = os.path.abspath('./data')
import crash_on_ipy
from multiprocessing import Pool
from collections import defaultdict
import pandas as pd

FEATURES = os.path.abspath('./features')

def parse(raw, root):
    def _consume(raw, i):
        for j in range(i, len(raw)):
            if raw[j] == ' ':
                return raw[i:j], j + 1

            if raw[j] == ')':
                return raw[i:j], j

    def _parse(raw, i, root):
        if raw[i] == ' ':
            i = i+1

        if raw[i] == '(':
            label, j = _consume(raw, i+1)
            root.label = label
            root.children = []
            root.isLeaf = False
            while raw[j] != ')':
                if raw[j] == '\n':
                    return j

                sub_root_idx = len(root.children)
                root.children.append(Tree(root.lam))
                j = _parse(raw, j, root.children[sub_root_idx])
                root.nnodes += root.children[sub_root_idx].nnodes
                root.weight += root.children[sub_root_idx].weight

            return j+1

        else:
            label, j = _consume(raw, i)
            root.isLeaf = True
            root.label = label
            root.weight = 1

            return j

    _parse(raw, 0, root)

class Tree(object):

    def __init__(self, lam):
        self.label = None
        self.children = []
        self.isLeaf = False
        self.nnodes = 1
        self.lam = lam
        self.weight = lam

    def get_nodes(self):
        res = []

        def add_nodes(t, nodes):
            nodes.append(t)
            for child in t.children:
                add_nodes(child, nodes)

        add_nodes(self, res)

        return res

class TreeKernel(object):

    def __init__(self, lam, nnodes):
        self.lam = lam
        self.nnodes = nnodes
        self.mem = np.zeros((nnodes, nnodes))
        self.index1 = {}
        self.index2 = {}

    def _init_mem(self, N1, N2):
        self.mem[:N1, :N2] = -1

    def index_of(self, nodes, node):
        for i in range(len(nodes)):
            if nodes[i] == node:
                return i

    def _norm(self, x):

        nodes = x.get_nodes()
        self._init_index(nodes, nodes)
        N = min(self.nnodes, len(nodes))

        self._init_mem(N, N)
        for i in range(N):
            self.mem[i, i] = nodes[i].weight

        res = 0.0

        for i in range(N):
            for j in range(N):
                res += self._compute(i, j, nodes, nodes)

        return np.sqrt(res)

    def _same_production(self, x, y):

        if x.label != y.label or len(x.children) != len(y.children):
            return False

        for ch1, ch2 in zip(x.children, y.children):
            if ch1.label != ch2.label:
                return False

        return True

    def _compute(self, i, j, nodes1, nodes2):

        if self.mem[i, j] >= 0:
            return self.mem[i, j]

        node_i = nodes1[i]
        node_j = nodes2[j]

        if node_i.label == node_j.label and \
                self._same_production(node_i, node_j):
            self.mem[i, j] = self.lam

            if (not node_i.isLeaf) and (not node_j.isLeaf):
                children1 = node_i.children
                children2 = node_j.children

                for k in range(0, min(len(children1), len(children2))):
                    self.mem[i, j] += self._compute(
                        self.index1[children1[k]],
                        self.index2[children2[k]],
                        nodes1, nodes2
                    )

            elif node_i.isLeaf and node_j.isLeaf:
                self.mem[i, j] = 1

        else:
            self.mem[i, j] = 0

        return self.mem[i, j]

    def _init_index(self, nodes1, nodes2):
        for i, node in enumerate(nodes1):
            self.index1[node]=i

        for i, node in enumerate(nodes2):
            self.index2[node]=i

    def cos(self, x, y):
        return self.eval(x, y)/(self._norm(x) * self._norm(y))

    def vec(self, x, y):

        self.eval(x, x)
        mem_x = np.sqrt(np.array(self.mem))
        self.eval(y, y)
        mem_y = np.sqrt(np.array(self.mem))

        res = defaultdict(float)

        nodes1 = x.get_nodes()
        nodes2 = y.get_nodes()
        self._init_index(nodes1, nodes2)

        self.N1 = min(self.nnodes, len(nodes1))
        self.N2 = min(self.nnodes, len(nodes2))
        self._init_mem(self.N1, self.N2)

        for i in range(self.N1):
            for j in range(self.N2):
                val = self._compute(i, j, nodes1, nodes2)
                if val > 0:
                    val_normed = (val+.0)/(mem_x[i, i] * mem_y[j, j])

                    if not nodes1[i].isLeaf:
                        res[nodes1[i].label] += val_normed
                    else:
                        res['LEAF'] += val_normed

        return res

    def eval(self, x, y):

        nodes1 = x.get_nodes()
        nodes2 = y.get_nodes()
        self._init_index(nodes1, nodes2)

        N1 = min(self.nnodes, len(nodes1))
        N2 = min(self.nnodes, len(nodes2))
        self._init_mem(N1, N2)


        res = 0.0

        for i in range(N1):
            for j in range(N2):
                res += self._compute(i, j, nodes1, nodes2)

        return res

class TreeKernel2(object):

    def __init__(self, lam, nnodes):
        self.lam = lam
        self.nnodes = nnodes
        self.mem = np.zeros((nnodes, nnodes))
        self.index1 = {}
        self.index2 = {}

    def _init_mem(self, N1, N2):
        self.mem[:N1, :N2] = -1

    def index_of(self, nodes, node):
        for i in range(len(nodes)):
            if nodes[i] == node:
                return i

    def _norm(self, x):

        nodes = x.get_nodes()
        self._init_index(nodes, nodes)
        N = min(self.nnodes, len(nodes))

        self._init_mem(N, N)
        for i in range(N):
            self.mem[i, i] = nodes[i].weight

        res = 0.0

        for i in range(N):
            for j in range(N):
                res += self._compute(i, j, nodes, nodes)

        return np.sqrt(res)

    def _same_production(self, x, y):

        if x.label != y.label or len(x.children) != len(y.children):
            return False

        for ch1, ch2 in zip(x.children, y.children):
            if ch1.label != ch2.label:
                return False

        return True

    def _compute(self, i, j, nodes1, nodes2):

        if self.mem[i, j] >= 0:
            return self.mem[i, j]

        node_i = nodes1[i]
        node_j = nodes2[j]

        if node_i.label == node_j.label and \
                self._same_production(node_i, node_j):

            self.mem[i, j] = self.lam * self.lam

            if (not node_i.isLeaf) and (not node_j.isLeaf):
                children1 = node_i.children
                children2 = node_j.children

                for k in range(0, min(len(children1), len(children2))):
                    self.mem[i, j] *= 1 + self._compute(
                        self.index1[children1[k]],
                        self.index2[children2[k]],
                        nodes1, nodes2
                    )
        else:
            self.mem[i, j] = 0

        return self.mem[i, j]

    def _init_index(self, nodes1, nodes2):
        for i, node in enumerate(nodes1):
            self.index1[node]=i

        for i, node in enumerate(nodes2):
            self.index2[node]=i

    def cos(self, x, y):
        xy = self.eval(x, y)
        # x_norm = self._norm(x)
        # y_norm = self._norm(y)
        x_norm = np.sqrt(self.eval(x, x))
        y_norm = np.sqrt(self.eval(y, y))
        return xy/(x_norm * y_norm)
        # return self.eval(x, y)/(self._norm(x) * self._norm(y))

    def vec(self, x, y):

        self.eval(x, x)
        mem_x = np.sqrt(np.array(self.mem))
        self.eval(y, y)
        mem_y = np.sqrt(np.array(self.mem))

        res = defaultdict(float)

        nodes1 = x.get_nodes()
        nodes2 = y.get_nodes()
        self._init_index(nodes1, nodes2)

        self.N1 = min(self.nnodes, len(nodes1))
        self.N2 = min(self.nnodes, len(nodes2))
        self._init_mem(self.N1, self.N2)

        for i in range(self.N1):
            for j in range(self.N2):
                val = self._compute(i, j, nodes1, nodes2)
                if val > 0:
                    val_normed = (val+.0)/(mem_x[i, i] * mem_y[j, j])

                    if not nodes1[i].isLeaf:
                        res[nodes1[i].label] += val_normed
                    else:
                        res['LEAF'] += val_normed

        return res


    def eval(self, x, y):

        nodes1 = x.get_nodes()
        nodes2 = y.get_nodes()
        self._init_index(nodes1, nodes2)

        self.N1 = min(self.nnodes, len(nodes1))
        self.N2 = min(self.nnodes, len(nodes2))
        self._init_mem(self.N1, self.N2)

        res = 0.0

        for i in range(self.N1):
            for j in range(self.N2):
                res += self._compute(i, j, nodes1, nodes2)

        return res

class TreeKernel3(object):

    def __init__(self, lam, nnodes):
        self.lam = lam
        self.nnodes = nnodes
        self.mem = np.zeros((nnodes, nnodes))
        self.index1 = {}
        self.index2 = {}

    def _init_mem(self, N1, N2):
        self.mem[:N1, :N2] = -1

    def index_of(self, nodes, node):
        for i in range(len(nodes)):
            if nodes[i] == node:
                return i

    def _norm(self, x):

        nodes = x.get_nodes()
        self._init_index(nodes, nodes)
        N = min(self.nnodes, len(nodes))

        self._init_mem(N, N)
        for i in range(N):
            self.mem[i, i] = nodes[i].weight

        res = 0.0

        for i in range(N):
            for j in range(N):
                res += self._compute(i, j, nodes, nodes)

        return np.sqrt(res)

    def _same_production(self, x, y):

        if x.label != y.label or len(x.children) != len(y.children):
            return False

        for ch1, ch2 in zip(x.children, y.children):
            if ch1.label != ch2.label:
                return False

        return True

    def _compute(self, i, j, nodes1, nodes2):

        if self.mem[i, j] >= 0:
            return self.mem[i, j]

        node_i = nodes1[i]
        node_j = nodes2[j]

        if node_i.label == node_j.label and \
                self._same_production(node_i, node_j):

            self.mem[i, j] = self.lam * self.lam

            if (not node_i.isLeaf) and (not node_j.isLeaf):
                children1 = node_i.children
                children2 = node_j.children

                for k in range(0, min(len(children1), len(children2))):
                    self.mem[i, j] *= 1 + self._compute(
                        self.index1[children1[k]],
                        self.index2[children2[k]],
                        nodes1, nodes2
                    )
        else:
            self.mem[i, j] = 0

        return self.mem[i, j]

    def _init_index(self, nodes1, nodes2):
        for i, node in enumerate(nodes1):
            self.index1[node]=i

        for i, node in enumerate(nodes2):
            self.index2[node]=i

    def cos(self, x, y):
        xy = self.eval(x, y)
        # x_norm = self._norm(x)
        # y_norm = self._norm(y)
        return xy
        # return self.eval(x, y)/(self._norm(x) * self._norm(y))

    def eval(self, x, y):

        nodes1 = x.get_nodes()
        nodes2 = y.get_nodes()
        self._init_index(nodes1, nodes2)

        N1 = min(self.nnodes, len(nodes1))
        N2 = min(self.nnodes, len(nodes2))
        self._init_mem(N1, N2)


        res = 0.0

        for i in range(N1):
            for j in range(N2):
                res += self._compute(i, j, nodes1, nodes2)

        return res

class TreeKernel4(object):

    def __init__(self, lam, nnodes):
        self.lam = lam
        self.nnodes = nnodes
        self.mem = np.zeros((nnodes, nnodes))
        self.index1 = {}
        self.index2 = {}

    def _init_mem(self, N1, N2):
        self.mem[:N1, :N2] = -1

    def index_of(self, nodes, node):
        for i in range(len(nodes)):
            if nodes[i] == node:
                return i

    def _norm(self, x):

        nodes = x.get_nodes()
        self._init_index(nodes, nodes)
        N = min(self.nnodes, len(nodes))

        self._init_mem(N, N)
        for i in range(N):
            self.mem[i, i] = nodes[i].weight

        res = 0.0

        for i in range(N):
            for j in range(N):
                res += self._compute(i, j, nodes, nodes)

        return np.sqrt(res)

    def _compute(self, i, j, nodes1, nodes2):

        if self.mem[i, j] >= 0:
            return self.mem[i, j]

        node_i = nodes1[i]
        node_j = nodes2[j]

        if node_i.label == node_j.label:
            self.mem[i, j] = self.lam

            if (not node_i.isLeaf) and (not node_j.isLeaf):
                children1 = node_i.children
                children2 = node_j.children

                for k in range(0, min(len(children1), len(children2))):
                    self.mem[i, j] += self._compute(
                        self.index1[children1[k]],
                        self.index2[children2[k]],
                        nodes1, nodes2
                    )
                    # self.mem[i, j] += self._compute(
                    #     self.index_of(nodes1, children1[k]),
                    #     self.index_of(nodes2, children2[k]),
                    #     nodes1, nodes2
                    # )

            elif node_i.isLeaf and node_j.isLeaf:
                self.mem[i, j] = 1

        else:
            self.mem[i, j] = 0

        return self.mem[i, j]

    def _init_index(self, nodes1, nodes2):
        for i, node in enumerate(nodes1):
            self.index1[node]=i

        for i, node in enumerate(nodes2):
            self.index2[node]=i

    def cos(self, x, y):
        return self.eval(x, y)

    def eval(self, x, y):

        nodes1 = x.get_nodes()
        nodes2 = y.get_nodes()
        self._init_index(nodes1, nodes2)

        N1 = min(self.nnodes, len(nodes1))
        N2 = min(self.nnodes, len(nodes2))
        self._init_mem(N1, N2)


        res = 0.0

        for i in range(N1):
            for j in range(N2):
                res += self._compute(i, j, nodes1, nodes2)

        return res

def progress_bar(percent, txt):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f} {:s}".format(
        "=" * fill, " " * (40 - fill), percent, txt), end='')

def load_parses(fpath_A, fpath_B):
    parses_A = []
    parses_B = []
    with open(fpath_A, 'r') as fa, \
        open(fpath_B, 'r') as fb:
        parses_A.extend(fa.readlines())
        parses_B.extend(fb.readlines())

    return parses_A, parses_B

    # tk = TreeKernel(1, 300)
    #
    # for i, (raw_A, raw_B) in enumerate(zip(raw_parses_A, raw_parses_B)):
    #     tree_A = Tree(lam = 1)
    #     tree_B = Tree(lam = 1)
    #     parse(raw_A, tree_A)
    #     parse(raw_B, tree_B)
    #     tk.cos(tree_A, tree_B)
    #     print((i+.0)/len(raw_parses_A))

def cal_vecs(fpath_A, fpath_B, lam = 0.5):
    # DATA = os.path.abspath('./data')
    # parse_Ax = os.path.join(DATA, 'parse_Ax.txt')
    # parse_Bx = os.path.join(DATA, 'parse_Bx.txt')
    parses_A, parses_B = load_parses(fpath_A, fpath_B)

    tk = TreeKernel2(lam, 300)
    cos_lst = []
    vec_lst = []
    pool = Pool()
    for i, (parse_A, parse_B) in enumerate(zip(parses_A, parses_B)):
        tree_A = Tree(lam=1)
        tree_B = Tree(lam=1)
        parse(parse_A, tree_A)
        parse(parse_B, tree_B)

        vec = tk.vec(tree_A, tree_B)
        vec_lst.append(vec)
        # print(vec)

        percent = (i +.0)/len(parses_A)
        progress_bar(percent, '')
        # print(i, (i + .0) / len(parses_A))

    df = pd.DataFrame(vec_lst)
    df = df.fillna(0)

    return df

def save(features):
    for name in features.columns:
        features.to_csv(os.path.join(FEATURES, '%s.txt' % name),
                             columns=[name], index=None, encoding='utf-8',
                             header=None)
    print(features.columns)

if __name__ == '__main__':
    DATA = os.path.abspath('./data')
    parse_Ax = os.path.join(DATA, 'parse_Ax.txt')
    parse_Bx = os.path.join(DATA, 'parse_Bx.txt')
    vecs = cal_vecs(parse_Ax, parse_Bx)
    save(vecs)


