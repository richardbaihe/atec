# -*-coding=utf-8 -*-
from __future__ import division
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier

class XGB():
    def __init__(self,model_name):
        self.model_name = model_name

    def train(self,train_final,val_final):
        val_label = val_final.pop('label')
        val_data = val_final
        train_label = train_final.pop('label')
        train_data = train_final
        # construct our model
        xgb_model = XGBClassifier(
            #        params,
            # eta=0.1,
            learning_rate=0.1,
            n_estimators=300,
            max_depth=7,
            min_child_weight=2,
            gamma=1,
            silent=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            #        scale_pos_weight=1
            seed=19931218
        )
        # find the suitable amounts of estimators
        xgb_model.fit(train_data, train_label, eval_set=[(val_data, val_label)], eval_metric='auc',
                      early_stopping_rounds=10)
        joblib.dump(xgb_model, self.model_name, protocol=2)

    def predict(self,test_data):
        model = joblib.load(self.model_name)
        result = model.predict_proba(test_data)[:, 1]
        return result