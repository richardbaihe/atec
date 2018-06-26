#!/usr/bin/env python
# coding:utf-8

__author__ = 'Yu Usami <yu2118@columbia.edu>'
__date__ = '$Sep 26, 2012'

import sys
import json
from collections import defaultdict
import os
import crash_on_ipy
import time

TRAIN = os.path.abspath('../data/ctb8.0/train')

"""
Probabilistic Context-Free Grammar Parser
Do syntactic parsing for input sentences based on PCFG.
"""

def read_counts(counts_file):
    """
    Read frequency counts from a file and return an iterator yields 
    each entity as a list. 
    """
    try:
        fi = open(counts_file, 'r')
    except IOError:
        sys.stderr.write('ERROR: Cannot open %s.\n' % counts_file)
        sys.exit(1)

    for line in fi:
        fields = line.strip().split(' ')
        yield fields # yields a list of fields


class PCFGParser():
    """
    Stores each count of nonterminal, binary rule and unary rule.
    Estimates rule parameters with these counts.
    Parses input sentences from stdin by using CKY algorithm.
    Outputs parsed trees in JSON format.
    """
    def __init__(self):
        self.nonterminal_counts = defaultdict(int)
        self.binary_rule_counts = defaultdict(int)
        self.LHS_to_binary_rules = defaultdict(set)
        self.unary_rule_counts = defaultdict(int)
        self.ter_to_rules = defaultdict(set)
        # self.unary_rule_counts[('NN', '<UNK>')] = 1
        # self.ter_to_rules['<UNK>'].add(('NN', '<UNK>'))


    def train(self, counts_file):
        """
        Read counts from a counts file, then store counts for each type:
        nonterminal, binary rule and unary rule.
        """
        for l in read_counts(counts_file):
            n, count_type, args = int(l[0]), l[1], l[2:]

            if n == 1:
                continue

            if count_type == 'NONTERMINAL':
                self.nonterminal_counts[args[0]] = n
            elif count_type == 'BINARYRULE':
                self.binary_rule_counts[tuple(args)] = n
                self.LHS_to_binary_rules[args[0]].add((args[0], args[1], args[2]))
            else: # UNARYRULE counts
                self.unary_rule_counts[tuple(args)] = n
                self.ter_to_rules[args[1]].add((args[0], args[1]))

    def q(self, x, y1, y2):
        """
        Return binary rule parameters for a rule such that x -> y1 y2.
        """
        return float(self.binary_rule_counts[x, y1, y2]) / self.nonterminal_counts[x]

    def q_unary(self, x, w):
        """
        Return unary rule parameters for a rule such that x -> w.
        """
        if self.nonterminal_counts[x] != 0:
            return float(self.unary_rule_counts[x, w]) / self.nonterminal_counts[x]
        else:
            return 0

    def parse(self, sentences):
        """
        Do syntactic parsing for sentences by using CKY algorithm.
        Write parsed trees to stdout in JSON format.
        """
        for s in sentences:
            s = s.strip()
            if s:
                print json.dumps(self.CKY(s.split(' ')))

    def CKY(self, x):
        """
        Implementation of CKY algorithm.
        Return a tree for a sentence x. It assumes that the grammar is in
        Chomsky normal form.
        """
        n = len(x) # length of sentence x
        pi = defaultdict(float) # DP table pi
        bp = {} # back pointers
        N = self.nonterminal_counts.keys() # set of nonterminals

        # Base case
        for i in xrange(n):
            pi[i, i, x[i]] = 1


            # for unary in self.ter_to_rules[x[i]]:
            #     if self.q_unary(unary[0], x[i]):
            #         pi[i, i, unary[0]] = self.q_unary(unary[0], x[i])
            #     else:
            #         pi[i, i, ]
            #
            # if sum([self.unary_rule_counts[X, x[i]] for X in N]) < 5: # if x[i] is infrequent word
            #     w = '_RARE_' # use _RARE_ insted of the actual word
            # else: # x[i] is not infrequent word
            #     w = x[i]
            # for X in N:
            #     pi[i, i, X] = self.q_unary(X, w) # if X -> x[i] not in the set of rules, assign 0
        
        # Recursive case
        # l is the scope size
        for l in xrange(1, n): 
            for i in xrange(n-l):
                j = i + l
                for X in N:
                    max_score = 0
                    args = None
                    for R in self.LHS_to_binary_rules[X]:
                        Y, Z = R[1:]
                        for s in xrange(i, j):
                            if pi[i, s, Y] and pi[s + 1, j, Z]: # calculate score if both pi entries have non-zero score
                                score = self.q(X, Y, Z) * pi[i, s, Y] * pi[s + 1, j, Z]
                                if max_score < score:
                                    max_score = score
                                    args = Y, Z, s
                    if max_score: # update DP table and back pointers
                        pi[i, j, X] = max_score
                        bp[i, j, X] = args

        # Return
        if pi[0, n-1, 'S']:
            return self.recover_tree(x, bp, 0, n-1, 'S')
        else: # if the tree does not have the start symbol 'S' as the root
            max_score = 0
            args = None
            # find a max_score rooted with NT other than 'S'
            for X in N:
                if max_score < pi[0, n-1, X]:
                    max_score = pi[0, n-1, X]
                    args = 0, n-1, X
            return self.recover_tree(x, bp, *args)

    def recover_tree(self, x, bp, i, j, X):
        """
        Return the list of the parsed tree with back pointers.
        """
        if i == j:
            return [X, x[i]]
        else:
            Y, Z, s = bp[i, j, X]
            return [X, self.recover_tree(x, bp, i, s, Y), 
                       self.recover_tree(x, bp, s+1, j, Z)]


def usage():
    print """Usage: python pcfg_parser.py [counts_file] < [input_file]

Read counts file to train a PCFG parser and parse sentences in input file"""

if __name__ == '__main__':
    # if len(sys.argv) != 2: # expect exactly one argument
    #     usage()
    #     sys.exit(2)

    # parser = PCFGParser() # initialize a PCFG parser
    # parser.train(sys.argv[1]) # train with a counts file
    # parser.parse(sys.stdin) # parse sentences from stdin

    parser = PCFGParser() # initialize a PCFG parser
    parser.train(os.path.join(TRAIN,'train.txt')) # train with a counts file
    # parser.parse([u'花呗/NN 的/DEG 安全/NN 没有/AD 验证/VV 成功/NN'.encode('utf-8')]) # parse sentences from stdin
    a = time.time()
    for i in range(10):
        parser.parse([u'NN DEG NN AD VV NN'.encode('utf-8')]) # parse sentences from stdin
    b = time.time()
    print(b-a)
