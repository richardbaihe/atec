import pandas as pd
import argparse
from feature import Feature
from model import XGB


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output

    test = pd.read_csv(in_path,sep='\t',header=None)
    fea = Feature(test)
    fea.ED_distance()
    fea.tfidf_share()
    fea.tfidf_sim()
    for name in fea.features.columns:
        if name not in ['1-share', '2-share', '4-share', '6-share', 'ed', 'tfidf_share',
                       '1-tfidf_share','3-tfidf_share']:
            fea.features.pop(name)
    test_data = fea.features

    model = XGB(model_name='xgb.model')
    result = model.predict(test_data)

    test[4] = result
    test.to_csv(out_path,index=None,header=None,sep='\t',columns=[0,4])
