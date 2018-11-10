import pandas as pd
import argparse,csv
from process import preprocess_char
from model import LM_transformer

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='char')
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--save_dir', type=str, default='save/')
parser.add_argument('--data_dir', type=str, default='data/char_AB_unk_test.tsv')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_iter', type=int, default=3)
parser.add_argument('--n_batch', type=int, default=128)
parser.add_argument('--max_grad_norm', type=int, default=1)
parser.add_argument('--lr', type=float, default=6.25e-5)
parser.add_argument('--lr_warmup', type=float, default=0.002)
parser.add_argument('--n_ctx', type=int, default=93)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--clf_pdrop', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--vector_l2', action='store_true')
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--afn', type=str, default='gelu')
parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
parser.add_argument('--encoder_path', type=str, default='data/char_vocab.txt')
parser.add_argument('--n_transfer', type=int, default=12)
parser.add_argument('--lm_coef', type=float, default=0.2)
parser.add_argument('--b1', type=float, default=0.9)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--e', type=float, default=1e-8)
parser.add_argument('--pre_load', type=bool, default=False)
parser.add_argument('--pos_weight', type=float, default=0.8)

parser.add_argument('--input',required=True)
parser.add_argument('--output',required=True)
args = parser.parse_args()
in_path = args.input
out_path = args.output

test = pd.read_csv(in_path,sep='\t',header=None,quoting=csv.QUOTE_NONE,names=['index','A','B'])
preprocess_char(test,args.encoder_path)
model = LM_transformer(args)
result = model.predict(args.data_dir)

test['label'] = result
test.to_csv(out_path, index=None, header=None, sep='\t', columns=['index', 'label'])