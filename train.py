# -*-coding=utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
from feature import Feature
from model import XGB
from sklearn.metrics import f1_score
import os

if __name__ =='__main__':

    # data = pd.read_csv('data/seg_Ax.txt', sep='\t', header=None, names=['seg_Ax'], encoding='utf-8', dtype=str)
    # data['seg_Bx'] = pd.read_csv('data/seg_Bx.txt', header=None, encoding='utf-8', dtype=str)
    # data['label'] = pd.read_csv('data/label.txt',header=None)
    data = pd.read_csv('data/seg_Axr.txt', sep='\t', header=None, names=['seg_Ax'], encoding='utf-8', dtype=str)
    data['seg_Bx'] = pd.read_csv('data/seg_Bxr.txt', header=None, encoding='utf-8', dtype=str)
    data['label'] = pd.read_csv('data/label_xr.txt',header=None)

    # fea = Feature(data,tr=True)
    fea = Feature(data,tr=False)
    #
    # fea.tfidf_sim([1,2,3])
    # print('tfidf_sim done.')
    # fea.ED_distance()
    # print('ED_distance done.')
    # fea.tfidf_share([1,3])
    # print('tfidf_share done.')
    # fea.ngram_share([1,2,4,6])
    # print('ngram_share done.')
    # fea.LSA_simlar()
    # print('LSA_simlar done.')
    # fea.LDA_simlar()
    # print('LDA_simlar done.')
    # fea.save()
    # exit()
    # print('parsing...')
    # os.system('java -jar jars/stanford_parser.jar data/%s data/%s data/%s data/%s'
    #           % ('seg_Axr.txt', 'seg_Bxr.txt', 'parses_A.txt', 'parses_B.txt'))
    # print('parsing done.')
    # fea.syntactic('data/parses_A.txt', 'data/parses_B.txt')
    # print('syntactic done.')
    # fea.save()
    fea.load()

    # valid_index = np.load('data/valid_index.npy')
    valid_index = np.load('data/valid_index_xr.npy')
    train_index = list(set(valid_index)^set(data.index))
    train_data = fea.features.iloc[train_index]
    valid_data = fea.features.iloc[valid_index]

    model = XGB(model_name='xgb.model')
    train_label = train_data.pop('label')
    valid_label = valid_data.pop('label')
    model.train(train_data,valid_data,train_label,valid_label)
    result = model.predict(valid_data)
    f1 = f1_score(valid_label.values,result)
    # ans[name] = f1
    print('F1_score of valid data:\t', f1)
    # for key,value in ans.items():
    #     print(key+'\t'+str(value))