import pandas as pd
import argparse
import jieba
import re
from feature import Feature
from model import XGB


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output

    data = pd.read_csv(in_path,sep='\t',header=None)
    data.columns = ['index', 'A', 'B']
    jieba.load_userdict("data/dict.txt")
    data['seg_A'] = data['A'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))
    data['seg_B'] = data['B'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))

    pattern = re.compile(r'\*+')
    data['Ax'] = data['A'].apply(lambda x: re.sub(pattern, '*', x))
    data['Bx'] = data['B'].apply(lambda x: re.sub(pattern, '*', x))

    data['seg_Ax'] = data['Ax'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))
    data['seg_Bx'] = data['Bx'].apply(lambda x: ' '.join(jieba.cut(x.strip(), cut_all=False)))

    fea = Feature(data)

    fea.tfidf_sim(3)
    print('tfidf_sim done.')
    fea.ED_distance()
    print('ED_distance done.')
    fea.tfidf_share(3)
    print('tfidf_share done.')
    fea.ngram_share(3)
    print('ngram_share done.')
    fea.LSA_simlar()
    print('LSA_simlar done.')
    fea.LDA_simlar()
    print('LDA_simlar done.')

    test_data = fea.features

    model = XGB(model_name='xgb.model')
    result = model.predict(test_data)

    data['pred'] = result
    data.to_csv(out_path,index=None,header=None,sep='\t',columns=['index','pred'])
