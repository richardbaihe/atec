from __future__ import division
import pandas as pd
import xgboost as xgb
import argparse,os
from sklearn.externals import joblib
from sklearn.metrics import auc
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier

features = ['adCategoryId','advertiserId','age','aid','campaignId','carrier','consumptionAbility','creativeId','creativeSize','education','gender','house','LBS','productId','productType']
chunks = []
temp_fea = features + ['label']
for name in temp_fea:
    print("mering "+name)
    chunk = pd.read_csv('../single_feature/'+name+'.csv',skip_blank_lines=False)
    chunks.append(chunk)
data = pd.concat(chunks,axis = 1)
train = data[data.label!=-1]
test = data[data.label==-1]
def submission(model_name):
    #test = pd.read_csv('~/tecent/test_all_features.csv')
    uid = test['uid'].astype(int)
    aid = test['aid'].astype(int)
    test_data = test[features]
    model = joblib.load(model_name)
    result = model.predict_proba(test_data)[:, 1]

    submission = pd.DataFrame({'aid':aid, 'uid': uid, 'score': result})
    #submission = submission.sort_values("aid", axis=0)
    submission.to_csv("submission.csv", columns=['aid','uid','score'],index=False)
def validation(model_name):
    model = joblib.load(model_name)
    #test = train[(train.clickTime < 31 * 1000000) & (train.clickTime >= 30 * 1000000)]
    test= train[train.index>=train.shape[0]*0.2]
    test_data = test[features]
    test_label = test['label']
    result = model.predict_proba(test_data)[:, 1]
    ans = auc(test_label,result)
    print (ans)
def training(model_name,useTrainCV=False,cv_folds=5):
    # prepare data for training and validating
    train_final = train[train.index<train.shape[0]*0.2]
    test_final = train[train.index>=train.shape[0]*0.2]
    #train_final = train[train.label==1]
    #test_final = train[train.label!=1]
    test_data = test_final[features]
    test_label = test_final['label']
    train_data = train_final[features]
    train_label = train_final['label']
    # construct our model
    params = {'max_depth': 7,
              'eta': 0.1,
              'gamma':1,
              'colsample_bylevel':0.8,
              'lambda':5,
              'silent': 1,
              'objective': 'binary:logistic',
              'eval_metric':'auc',
              'min_child_weight':2,
              'subsample':0.8,
              'colsample_bytree':0.8,
              'seed':19931218
              }
    xgb_model = XGBClassifier(
#        params,
        #eta=0.1,
        learning_rate=0.1,
        n_estimators=300,
        max_depth=7,
        min_child_weight=2,
        gamma=1,
        silent =1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
#        scale_pos_weight=1
        seed=19931218
    )
    # find the suitable amounts of estimators
    if(useTrainCV):
        xgb_param = xgb_model.get_xgb_params()
        xgtrain = xgb.DMatrix(train_data.values,label=train_label.values)
        cvresult = xgb.cv(xgb_param, xgtrain,num_boost_round=xgb_model.get_params()['n_estimators'],
                          nfold=cv_folds,metrics='logloss',early_stopping_rounds=20,show_stdv=False)
        xgb_model.set_params(n_estimators= cvresult.shape[0])
    xgb_model.fit(train_data, train_label, eval_set=[(test_data, test_label)], eval_metric='auc', early_stopping_rounds=10)
    joblib.dump(xgb_model, model_name,protocol=2)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose a mode from train, validation, submission like: python train_model train')
    parser.add_argument('-mode',type=str,default='train')
    parser.add_argument('-model_name', type=str, default='all_data')
    args = parser.parse_args()
    if not os.path.exists('./model'):
      os.mkdir('./model')
    model_name = "model/"+ args.model_name +".pkl"
    if args.mode=='train':
        training(model_name)
        submission(model_name)
    elif args.mode=='validation':
        validation(model_name)
    else:
        submission(model_name)
