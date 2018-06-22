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

    fea = Feature(data,tr=True)

    # fea.LDA_simlar()
    fea.tfidf_sim(6)
    fea.ED_distance()
    fea.tfidf_share(6)
    fea.ngram_share(6)

    for name in fea.features.columns:
       fea.features.to_csv('fea/'+name+'.csv', columns=[name],index=None)

    for name in ['1-share','2-share','4-share','6-share','ed','tfidf_share',
                 '1-tfidf_share','3-tfidf_share']:
        fea.features[name] = pd.read_csv('fea/'+name+'.csv')


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
    # ans[name] = f1
    print('F1_score of valid data:\t', f1)
    # for key,value in ans.items():
    #     print(key+'\t'+str(value))