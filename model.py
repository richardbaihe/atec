# -*-coding=utf-8 -*-
from __future__ import division
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score

def xg_f1(y,t):
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y] # binaryzing your output
    return 'f1',f1_score(t,y_bin)

class XGB():
    def __init__(self,model_name):
        self.model_name = model_name

    def train(self,train_data,val_data,train_label,val_label):
        # construct our model
        xgb_model = XGBClassifier(
            #        params,
            # eta=0.1,
            learning_rate=0.001,
            n_estimators=10000,
            max_depth=5,
            min_child_weight=2,
            gamma=0.2,
            silent=1,
            subsample=0.9,
            colsample_bytree=0.7,
            objective='binary:logistic',
            # scale_pos_weight=3.99,
            # max_delta_step=0.65,
            seed=27
        )
        # find the suitable amounts of estimators
        xgb_model.fit(train_data, train_label, eval_set=[(val_data, val_label)], eval_metric=xg_f1,
                      early_stopping_rounds=50)
        joblib.dump(xgb_model, self.model_name, protocol=2)

    def predict(self,test_data):
        model = joblib.load(self.model_name)
        result = model.predict(test_data)
        return result
