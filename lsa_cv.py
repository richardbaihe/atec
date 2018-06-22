# coding:utf-8
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt
import os
import pandas as pd


data = pd.read_csv('data/seg_Ax.txt', sep='\t', header=None, names=['seg_Ax'], encoding='utf-8', dtype=str)
data['seg_Bx'] = pd.read_csv('data/seg_Bx.txt', header=None, encoding='utf-8', dtype=str)
data['label'] = pd.read_csv('data/label.txt',header=None)

df_texts = pd.concat([data['seg_Ax'], data['seg_Bx']])
texts = df_texts.apply(lambda x: [word for word in x.split()]).values
feature = pd.DataFrame()
feature['label'] = data['label']

corpus = pd.concat([data['seg_Ax'],data['seg_Bx']])
cntVector = CountVectorizer()
cntTf = cntVector.fit_transform(corpus)

parameters = {'learning_method':('batch', 'online'),
              'n_topics':(10, ),
              'doc_topic_prior':(0.001, 0.01, 0.1),
              'topic_word_prior':(0.001, 0.01, 0.1),
              # 'doc_topic_prior':(0.001, 0.01, 0.05, 0.1, 0.2),
              # 'topic_word_prior':(0.001, 0.01, 0.05, 0.1, 0.2),
              'max_iter':(20, )}
lda = LatentDirichletAllocation()
model = GridSearchCV(estimator=lda, param_grid=parameters,n_jobs=3)
model.fit(cntTf)
print(model.grid_scores_, model.best_params_, model.best_score_)
print(sorted(model.cv_results_.keys()))