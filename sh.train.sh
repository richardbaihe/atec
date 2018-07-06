#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python -u train.py --n_gpu=1 --n_embd=384 --n_ctx=93 --n_batch=64 --n_iter=15 --lr=1e-4 --n_head=12 --n_layer=6 --lm_coef=0.5 --data_dir=data/para.tsv --encoder_path=data/char_vocab.txt --desc=paraphrase
