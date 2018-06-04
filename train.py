import pandas as pd


if __name__ =='__main__':

    data = pd.read_csv('data/seg_A.txt',sep='\t',header=None)
    fea = Feature(test)
    test_feature = feature(test)
    result = predict(test_feature)
    test[4] = result
    test.to_csv(out_path,index=None,header=None,sep='\t',columns=[0,4])
