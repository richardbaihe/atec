# -*-coding=utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from gensim import models,corpora
from gensim.models.phrases import Phraser
import scipy,codecs,six,re,os
from scipy.spatial import distance
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import TfidfModel
from collections import Counter,defaultdict
import jieba
from sklearn.externals import joblib
import crash_on_ipy
import tree_kernel

FEATURES = os.path.abspath('./features')
DATA = os.path.abspath('./data')
N = 3
# The order of the feature names matter!
SYN  =  ['AD', 'ADJP', 'ADVP', 'AS', 'BA', 'CC', 'CD', 'CLP', 'CP',
         'CS', 'DEC', 'DEG', 'DEV', 'DNP', 'DP', 'DT', 'DVP', 'ETC',
         'FRAG', 'IP', 'JJ', 'LB', 'LC', 'LCP', 'LEAF', 'LST', 'M',
         'MSP', 'NN', 'NP', 'NR', 'NT', 'OD', 'P', 'PN', 'PP', 'PRN',
         'PU', 'QP', 'ROOT', 'SB', 'SP', 'UCP', 'VA', 'VC', 'VCD',
         'VCP', 'VE', 'VP', 'VRD', 'VSB', 'VV']

# NAMES = [str(i)+'-tfidf_sim' for i in range(N+1)]+\
#         ['ed']+\
#         [str(i)+'-tfidf_share' for i in range(N+1)]+\
#         [str(i)+'-share' for i in range(N+1)]+\
#         ['lsa_sim','lda_sim']+ \
#         SYN
LEX = ['1-share','2-share','4-share','6-share','ed',
         '1-tfidf_share','1-tfidf_sim','3-tfidf_sim',
         '3-tfidf_share','2-tfidf_sim','lda_sim','lsa_sim']

NAMES = LEX \
        + SYN


LABEL = 'label'

class Feature():
    def __init__(self,data,tr=False):
        # stopwords
        stpwrdpath = "data/stop_words"
        self.stpwrdlst = []
        # if six.PY2:
        #     for line in open(stpwrdpath, 'r'):
        #         word = line.strip().decode('gbk')
        #         self.stpwrdlst.append(word)
        # else:
        #     for line in open(stpwrdpath, 'r',encoding='gbk'):
        #         word = line.strip()
        #         self.stpwrdlst.append(word)
        # word2index
        # dic = {}
        # for index, line in enumerate(codecs.open('data/vocab.txt', 'r', encoding='utf-8')):
        #     word, freq = line.split()
        #     if int(freq) <= 5:
        #         self.stpwrdlst.append(word)
        #     dic[word] = index
        self.tr=tr
        # if not tr:
        #     data.columns = ['index', 'A', 'B']
        #     jieba.load_userdict("data/dict.txt")
        #     data['seg_A'] = data['A'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))
        #     data['seg_B'] = data['B'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))
        #
        #     pattern = re.compile(r'\*+')
        #     data['Ax'] = data['A'].apply(lambda x: re.sub(pattern, '*', x))
        #     data['Bx'] = data['B'].apply(lambda x: re.sub(pattern, '*', x))
        #
        #     data['seg_Ax'] = data['Ax'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))
        #     data['seg_Bx'] = data['Bx'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))

        self.df_texts = pd.concat([data['seg_Ax'], data['seg_Bx']])
        texts = self.df_texts.apply(lambda x: [word for word in x.split()
                                               if word not in self.stpwrdlst]).values
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        self.texts = [[token for token in text if frequency[token]>1] for text in texts]
        self.data = data
        self.features = pd.DataFrame()
        if self.tr:
            self.features['label'] = data.label

    def LDA_simlar(self):
        corpus = pd.concat([self.data['seg_Ax'], self.data['seg_Bx']])
        cntVector = CountVectorizer(stop_words=self.stpwrdlst)
        # cntTf = cntVector.fit_transform(corpus)

        lda = LatentDirichletAllocation(n_topics=10,
                                        learning_offset=50.,
                                        random_state=0)

        docres = None

        if not self.tr:
            cntVector = joblib.load('model/LDA_tf.model')
            cntTf = cntVector.transform(corpus)
            lda = joblib.load('model/LDA.model')
        else:
            cntTf = cntVector.fit_transform(corpus)
            joblib.dump(cntVector, 'model/LDA_tf.model')
            lda.fit(cntTf)
            joblib.dump(lda, 'model/LDA.model')

        docres = lda.transform(cntTf)

        lda_q1 = docres[:docres.shape[0] // 2]
        lda_q2 = docres[docres.shape[0] // 2:]

        lda_sim = pd.DataFrame([distance.cosine(x, y) for x, y in zip(lda_q1, lda_q2)])
        self.features['lda_sim'] = lda_sim

    def LSA_simlar(self):
        corpus = pd.concat([self.data['seg_Ax'], self.data['seg_Bx']])
        cntVector = CountVectorizer(stop_words=self.stpwrdlst)
        # cntTf = cntVector.fit_transform(corpus)

        lsa = TruncatedSVD(n_components=400, random_state=0)
        docres = None

        if not self.tr:
            cntVector = joblib.load('model/LSA_tf.model')
            cntTf = cntVector.transform(corpus)
            lsa = joblib.load('model/LSA.model')
        else:
            cntTf = cntVector.fit_transform(corpus)
            lsa.fit(cntTf)
            joblib.dump(lsa, 'model/LSA.model')
            joblib.dump(cntVector, 'model/LSA_tf.model')

        docres = lsa.transform(cntTf)
        lsa_q1 = docres[:docres.shape[0] // 2]
        lsa_q2 = docres[docres.shape[0] // 2:]

        lsa_sim = pd.DataFrame([distance.cosine(x, y) for x, y in zip(lsa_q1, lsa_q2)])
        # lsa_sim = pd.DataFrame([np.abs(x - y) for x, y in zip(lsa_q1, lsa_q2)])
        self.features['lsa_sim'] = lsa_sim

    def ED_distance(self):
        # def edit_distance(row):
        #     q1words = {}
        #     q2words = {}
        #     for word in ''.join(row['seg_Ax'].lower().split()):
        #         if word not in self.stpwrdlst:
        #             q1words[word] = 1
        #     for word in ''.join(row['seg_Bx'].lower().split()):
        #         if word not in self.stpwrdlst:
        #             q2words[word] = 1
        #     if len(q1words) == 0 or len(q2words) == 0:
        #         # The computer-generated chaff includes a few questions that are nothing but stopwords
        #         return max(len(q1words), len(q2words))
        #     a_sp = list(q1words.keys())
        #     b_sp = list(q2words.keys())
        #     dp = [[0 for _ in range(len(a_sp) + 1)] for __ in range(len(b_sp) + 1)]
        #     dp[0] = [i for i in range(len(a_sp) + 1)]
        #     for i in range(len(b_sp) + 1):
        #         dp[i][0] = i
        #     for i in range(len(b_sp)):
        #         for j in range(len(a_sp)):
        #             if a_sp[j] == b_sp[i]:
        #                 temp = 0
        #             else:
        #                 temp = 1
        #             dp[i + 1][j + 1] = min(dp[i][j] + temp, dp[i][j + 1] + 1, dp[i + 1][j] + 1)
        #     return dp[-1][-1]

        def edit_distance2(row):

            def ed(s1, s2):
                m = len(s1) + 1
                n = len(s2) + 1
                i = None
                j = None
                tbl = {}
                for i in range(m): tbl[i, 0] = i
                for j in range(n): tbl[0, j] = j
                for i in range(1, m):
                    for j in range(1, n):
                        cost = 0 if s1[i - 1] == s2[j - 1] else 1
                        tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)

                return tbl[i, j]

            return ed(row['seg_Ax'].split(), row['seg_Bx'].split())

        train_edit_distance = self.data.apply(edit_distance2, axis=1, raw=True)
        self.features['ed'] = train_edit_distance

    def ngram_share(self, n_lst = []):
        def word_match_share(q1words,q2words):
            shared_words_in_q1 = [w for w in q1words if w in q2words]
            shared_words_in_q2 = [w for w in q2words if w in q1words]
            R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
            return R

        cur_gram = self.texts
        for i in range(1, max(n_lst)+1):
            cur_gram_q1 = cur_gram[:len(cur_gram) // 2]
            cur_gram_q2 = cur_gram[len(cur_gram) // 2:]
            if i in n_lst:
                self.features[str(i)+'-share'] = [word_match_share(x,y) for x,y in
                                        zip(cur_gram_q1,cur_gram_q2)]
            if not self.tr:
                next_gram_model = models.Phrases.load('model/'+str(i)+'-share.model')
            else:
                phrases = models.Phrases(cur_gram)
                next_gram_model = Phraser(phrases)
                if not os.path.exists('model/'+str(i)+'-share.model'):
                    next_gram_model.save('model/'+str(i)+'-share.model')
            next_gram = next_gram_model[cur_gram]
            cur_gram = next_gram

    def tfidf_share(self, n_lst = []):
        # If a word appears only once, we ignore it completely (likely a typo)
        # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
        def get_weight(count, eps=10000, min_count=2):
            if count < min_count:
                return 0
            else:
                return 1.0 / (count + eps)

        def tfidf_word_match_share(q1words,q2words,weights):
            shared_weights = [weights.get(w, 0) for w in q1words if w in q2words] + [weights.get(w, 0) for w in
                                                                                            q2words if
                                                                                            w in q1words]
            total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
            R = np.sum(shared_weights) / np.sum(total_weights)
            return R
        cur_gram = self.df_texts
        for i in range(1, max(n_lst)+1):
            words = (" ".join(cur_gram)).lower().split()
            counts = Counter(words)
            weights = {word: get_weight(count) for word, count in counts.items()}

            cur_gram_q1 = cur_gram[:len(cur_gram) // 2]
            cur_gram_q2 = cur_gram[len(cur_gram) // 2:]
            if i in n_lst:
                self.features[str(i)+'-tfidf_share'] = [tfidf_word_match_share(x,y,weights) for x,y in
                                        zip(cur_gram_q1,cur_gram_q2)]
            if not self.tr:
                next_gram_model = models.Phrases.load('model/'+str(i)+'-tfidf-share.model')
            else:
                phrases = models.Phrases(cur_gram)
                next_gram_model = Phraser(phrases)
                if not os.path.exists('model/'+str(i)+'-tfidf-share.model'):
                    next_gram_model.save('model/'+str(i)+'-tfidf-share.model')
            next_gram = next_gram_model[cur_gram]
            cur_gram = next_gram

    def tfidf_sim(self,n_lst = []):
        def cosine(a, b):
            sum = 0
            for key in a.keys():
                if key in b.keys():
                    sum += a[key] * b[key]
            return 0.5 + 0.5 * sum
        cur_gram = self.df_texts
        for i in range(1, max(n_lst)+1):
            dictionary = corpora.Dictionary([line.split(' ') for line in  cur_gram])
            corpus = [dictionary.doc2bow(line.split(' ')) for line in cur_gram]
            if not self.tr:
                model = TfidfModel.load('model/' + str(i) + '-tfidf.model')
            else:
                model = TfidfModel(corpus)
                model.save('model/'+str(i)+'-tfidf.model')

            tfidf = model[corpus]

            tfidf_q1 = tfidf[:len(tfidf) // 2]
            tfidf_q2 = tfidf[len(tfidf) // 2:]
            tfidf_sim = [cosine(dict(x), dict(y)) for x, y in
                         zip(tfidf_q1, tfidf_q2)]
            if i in n_lst:
                self.features[str(i)+'-tfidf_sim'] = tfidf_sim
            if not self.tr:
                next_gram_model = models.Phrases.load('model/'+str(i)+'-share.model')
            else:
                phrases = models.Phrases(cur_gram)
                next_gram_model = Phraser(phrases)
                if not os.path.exists('model/'+str(i)+'-share.model'):
                    next_gram_model.save('model/'+str(i)+'-share.model')
            next_gram = next_gram_model[cur_gram]
            cur_gram = next_gram

    def syntactic(self, parses_Ax, parses_Bx):
        vecs = tree_kernel.cal_vecs(parses_Ax, parses_Bx)
        # for name in vecs.columns:
        #     self.features[name] = vecs[name]
        n = len(parses_Ax)
        for name in SYN:
            if name in vecs.columns:
                self.features[name] = vecs[name]
            else:
                self.features[name] = np.zeros(len(vecs))

    def save(self):
        for name in self.features.columns:
            if name == 'label':
                continue

            self.features.to_csv(os.path.join(FEATURES, '%s.txt' % name),
                                 columns=[name], index=None, encoding='utf-8',
                                 header=None)

    def load(self):
        self.features = pd.read_csv(os.path.join(DATA, 'label_xr.txt'),
                               sep='\t', header=None, names=['label'], encoding='utf-8', dtype=float)
        for name in NAMES:
        # for name in LEX:
            fpath = os.path.join(FEATURES, '%s.txt' % name)
            # if os.path.exists(fpath):
            self.features[name] = \
                    pd.read_csv(fpath,
                                sep='\t', header=None, names=[name], encoding='utf-8', dtype=float)

            # else:
            #     self.features[name] = []

        # self.features = self.features.fillna(0)