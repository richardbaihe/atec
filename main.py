import pandas as pd
import argparse
from feature import Feature
def feature(data):
    '''
    :param data: DataFrame of original data
    :return: new DataFrame with classical features
    '''


    return data

def predict(data):
    '''
    :param data: DataFrame of features
    :return: Series predict results
    '''
    return 1

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output

    test = pd.read_csv(in_path,sep='\t',header=None)
fea = Feature(test)
    test_feature = feature(test)
    result = predict(test_feature)
    test[4] = result
    test.to_csv(out_path,index=None,header=None,sep='\t',columns=[0,4])
