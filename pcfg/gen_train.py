# coding:utf-8
from os import listdir
import os
from nltk.corpus import BracketParseCorpusReader, LazyCorpusLoader
from collections import defaultdict
import crash_on_ipy

CTB = os.path.abspath('../data/ctb8.0/data')
OUT = os.path.abspath('../data/ctb8.0/out')
TRAIN = os.path.abspath('../data/ctb8.0/train')

nonterminal_counts = defaultdict(int)
binary_rule_counts = defaultdict(int)
unary_rule_counts = defaultdict(int)


def convert():
    bracketed = os.path.join(CTB, 'bracketed')
    for fname in listdir(bracketed):
        with open(os.path.join(bracketed, fname), 'r') as src, \
                open(os.path.join(OUT, fname), 'w') as out:

            try:
                for line in src:
                    if (not line.startswith('<')) \
                            and len(line.strip()) > 0:
                        out.write(line)
            except:
                pass

class MyTree(object):

    def __init__(self, children, label):
        self._label = label
        self._children = children

    def __getitem__(self, i):
        return self._children[i]

    def __len__(self):
        return len(self._children)

def rule_counts(t):

    t._label = 'S'

    def get_rule_counts(t):

        # if type(t) != unicode:
        #     nonterminal_counts[t._label] += 1

        if len(t) == 1:
            if type(t[0]) == unicode:
                # means a leaf node
                unary_rule_counts[(t._label, t[0])] += 1
                # return
            else:
                # means t -> t[0], t[0] -> ...
                # new_label = '.'.join([t._label, t[0]._label])
                # t._label = new_label
                # t[0]._label = new_label
                # t[0]._label = t._label
                get_rule_counts(t[0])
                t._label = t[0]._label

        elif len(t) == 2:
            # means a binary rule
            get_rule_counts(t[0])
            get_rule_counts(t[1])
            # add rule count when backtracking for
            #  there might be tree node label changed
            binary_rule_counts[(t._label, t[0]._label, t[1]._label)] += 1

        elif len(t) > 2:
            # means t -> t[0] t[1] t[2] ...
            # make it into t -> t[0] X
            # X -> t[1] ...
            X = t._label
            if '*' not in X:
                X += '*'
            # X = '\\'.join([t._label, t[0]._label])
            # X = '+'.join([t[i]._label if type(t[i]) != unicode else t[i]
            #               for i in range(1, len(t))])
            X_children = [t[i] if type(t[i]) != unicode else MyTree([t[i]],t[i])
                          for i in range(1, len(t))]
            get_rule_counts(t[0])
            get_rule_counts(MyTree(X_children, X))
            # add rule count when backtrackingfor
            #  there might be tree node label changed
            binary_rule_counts[(t._label, t[0]._label, X)] += 1

        else:
            print('error')

        if type(t) != unicode and \
                not (len(t) == 1 and type(t[0]) != unicode):
            nonterminal_counts[t._label] += 1

    while len(t) == 1 and type(t[0]) != unicode:
        t = t[0]
        t._label = 'S'

    get_rule_counts(t)

def test(ctb):
    sents = ctb.parsed_sents(os.path.join(OUT, 'chtb_1141.mz'))
    for s in sents:
        rule_counts(s)
        break
    print('NTs:')
    for rule, count in nonterminal_counts.items():
        print(rule, count)

    print('unary\'s:')
    for rule, count in unary_rule_counts.items():
        print(rule, count)

    print('binary\'s:')
    for rule, count in binary_rule_counts.items():
        print(rule, count)
    exit()

if __name__ == '__main__':
    # convert()
    ctb = LazyCorpusLoader('ctb', BracketParseCorpusReader,
                           r'chtb_*.*', tagset='unknown')

    bracketed = os.path.join(CTB, 'bracketed')
    files = [os.path.join(OUT, fname) for fname in listdir(bracketed)]

    sents_lst = []
    for i, f in enumerate(files):
        # if i> 100:
        #     break
        sents = ctb.parsed_sents(f)
        sents_lst.append(sents)
        print('%d/%d\n' % (i, len(files)))

    for i, sents in enumerate(sents_lst):
        if i ==78 :
            print('aaa')
        for s in sents:
            rule_counts(s)
            print('%d/%d\n' % (i, len(files)))

    with open(os.path.join(TRAIN, 'train.txt'), 'w') as out:
        dicts = [nonterminal_counts, unary_rule_counts, binary_rule_counts]
        dict_names = ['NONTERMINAL', 'UNARYRULE', 'BINARYRULE']
        for dname, dict in zip(dict_names, dicts) :
            for rule, count in dict.items():
                if type(rule) == tuple:
                    rule = ' '.join(rule).encode('utf-8')
                out.write('%d %s %s\n' % (count, dname, rule))


