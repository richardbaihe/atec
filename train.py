import pandas as pd
from feature import Feature

if __name__ =='__main__':

    data = pd.read_csv('data/seg_A.txt',sep='\t',header=None,name='seg_A')
    data['seg_B'] = pd.read_csv('data/seg_B.txt')
    fea = Feature(data)
    test_feature = feature(test)
    result = predict(test_feature)
    test[4] = result
    test.to_csv(out_path,index=None,header=None,sep='\t',columns=[0,4])
