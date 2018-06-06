# -*-coding=utf-8 -*-
import pandas as pd
import numpy as np
from feature import Feature
from model import XGB

if __name__ =='__main__':

    data = pd.read_csv('data/seg_Ax.txt', sep='\t', header=None, names=['seg_Ax'], encoding='utf-8', dtype=str)
    data['seg_Bx'] = pd.read_csv('data/seg_Bx.txt', header=None, encoding='utf-8', dtype=str)
    data['label'] = pd.read_csv('data/label.txt',header=None)

    fea = Feature(data)
    fea.ED_distance()
    fea.tfidf_share()
    fea.tfidf_sim()

    valid_index = np.load('data/valid_index.npy')
    train_index = list(set(valid_index)^set(data.index))
    train_data = fea.features[train_index]
    valid_data = fea.features.iloc[valid_index]

    model = XGB(model_name='model.xgb')
    model.train(train_data,valid_data)