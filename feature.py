# -*-coding=utf-8 -*-
import pandas as pd
import numpy as np
import scipy
from scipy.spatial import distance
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import codecs
import six

class Feature():
    def __init__(self,data):
        # stopwords
        stpwrdpath = "data/stop_words"
        self.stpwrdlst = []
        if six.PY2:
            for line in open(stpwrdpath, 'r'):
                word = line.strip().decode('gbk')
                self.stpwrdlst.append(word)
        else:
            for line in open(stpwrdpath, 'r',encoding='gbk'):
                word = line.strip()
                self.stpwrdlst.append(word)
        # word2index
        dic = {}
        for index, line in enumerate(codecs.open('data/vocab.txt', 'r', encoding='utf-8')):
            word, freq = line.split()
            if int(freq) <= 5:
                self.stpwrdlst.append(word)
            dic[word] = index
        # data['q1'] = data['seg_Ax'].apply(lambda x: map(lambda y: dic[y], x.split()))
        # data['q2'] = data['seg_Bx'].apply(lambda x: map(lambda y: dic[y], x.split()))
        self.data = data
        self.features = pd.DataFrame()
        if 'label' in data.columns:
            self.features['label'] = data.label

    def LDA_simlar(self):
        corpus = pd.concat([self.data['seg_Ax'], self.data['seg_Bx']])
        cntVector = CountVectorizer(stop_words=self.stpwrdlst)
        cntTf = cntVector.fit_transform(corpus)

        lda = LatentDirichletAllocation(n_topics=100,
                                        learning_offset=50.,
                                        random_state=0)

        docres = lda.fit_transform(cntTf)

        lda_q1 = docres[:docres.shape[0] / 2]
        lda_q2 = docres[docres.shape[0] / 2:]

        lda_sim = pd.DataFrame([distance.cosine(x, y) for x, y in zip(lda_q1, lda_q2)])
        self.features['lda_sim'] = lda_sim

    def ED_distance(self):
        def edit_distance(row):
            q1words = {}
            q2words = {}
            for word in row['seg_Ax'].lower().split():
                if word not in self.stpwrdlst:
                    q1words[word] = 1
            for word in row['seg_Bx'].lower().split():
                if word not in self.stpwrdlst:
                    q2words[word] = 1
            if len(q1words) == 0 or len(q2words) == 0:
                # The computer-generated chaff includes a few questions that are nothing but stopwords
                return max(len(q1words), len(q2words))
            a_sp = list(q1words.keys())
            b_sp = list(q2words.keys())
            dp = [[0 for _ in range(len(a_sp) + 1)] for __ in range(len(b_sp) + 1)]
            dp[0] = [i for i in range(len(a_sp) + 1)]
            for i in range(len(b_sp) + 1):
                dp[i][0] = i
            for i in range(len(b_sp)):
                for j in range(len(a_sp)):
                    if a_sp[j] == b_sp[i]:
                        temp = 0
                    else:
                        temp = 1
                    dp[i + 1][j + 1] = min(dp[i][j] + temp, dp[i][j + 1] + 1, dp[i + 1][j] + 1)
            return dp[-1][-1]

        train_edit_distance = self.data.apply(edit_distance, axis=1, raw=True)
        self.features['ed'] = train_edit_distance

    def tfidf_share(self):
        corpus = pd.concat([self.data['seg_Ax'], self.data['seg_Bx']])

        # If a word appears only once, we ignore it completely (likely a typo)
        # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
        def get_weight(count, eps=10000, min_count=2):
            if count < min_count:
                return 0
            else:
                return 1.0 / (count + eps)

        words = (" ".join(corpus)).lower().split()
        counts = Counter(words)
        weights = {word: get_weight(count) for word, count in counts.items()}

        def tfidf_word_match_share(row):
            q1words = {}
            q2words = {}
            for word in row['seg_Ax'].lower().split():
                if word not in self.stpwrdlst:
                    q1words[word] = 1
            for word in row['seg_Bx'].lower().split():
                if word not in self.stpwrdlst:
                    q2words[word] = 1
            if len(q1words) == 0 or len(q2words) == 0:
                # The computer-generated chaff includes a few questions that are nothing but stopwords
                return 0

            shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                            q2words.keys() if
                                                                                            w in q1words]
            total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

            R = np.sum(shared_weights) / np.sum(total_weights)
            return R

        tfidf_train_word_match = self.data.apply(tfidf_word_match_share, axis=1, raw=True)
        self.features['tfidf_share'] = tfidf_train_word_match

    def tfidf_sim(self):
        corpus = pd.concat([self.data['seg_Ax'], self.data['seg_Bx']])

        vector = TfidfVectorizer(stop_words=self.stpwrdlst)
        tfidf = vector.fit_transform(corpus)
        tfidf_q1 = tfidf[:tfidf.shape[0] // 2]
        tfidf_q2 = tfidf[tfidf.shape[0] // 2:]

        tfidf_sim = [distance.cosine(x, y) for x, y in
                     zip(scipy.sparse.csr_matrix.todense(tfidf_q1), scipy.sparse.csr_matrix.todense(tfidf_q2))]

        #tfidf_sim = pd.DataFrame(tfidf_sim)
        self.features['tfidf_sim']=tfidf_sim