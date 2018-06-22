# -*-coding=utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
from feature import Feature
from model import XGB
from sklearn.metrics import f1_score

if __name__ =='__main__':

    data = pd.read_csv('data/seg_Ax.txt', sep='\t', header=None, names=['seg_Ax'], encoding='utf-8', dtype=str)
    data['seg_Bx'] = pd.read_csv('data/seg_Bx.txt', header=None, encoding='utf-8', dtype=str)
    data['label'] = pd.read_csv('data/label.txt',header=None)

    fea = Feature(data)
    fea.load()
    print(fea.features)
    exit()

    # fea.LDA_simlar()
    # print('LDA done.')
    fea.ED_distance()
    print('ED done.')
    fea.tfidf_share()
    print('tfidf_share done.')

    fea.save()
    exit()

    fea.tfidf_sim()
    print('tfidf_sim done.')
    fea.LSA_simlar()
    print('LSA done.')
    fea.words_overlap()
    print('words_overlap done.')
    fea.ngram_simlar(n=1)
    print('ngram_simlar done.')

    valid_index = np.load('data/valid_index.npy')
    train_index = list(set(valid_index)^set(data.index))
    train_data = fea.features.iloc[train_index]
    valid_data = fea.features.iloc[valid_index]

    model = XGB(model_name='xgb.model')
    train_label = train_data.pop('label')
    valid_label = valid_data.pop('label')
    model.train(train_data,valid_data,train_label,valid_label)

    result = model.predict(valid_data)
    f1 = f1_score(valid_label.values,result)

    print('F1_score of valid data:\t', f1)
