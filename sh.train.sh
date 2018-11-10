#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -u train.py --n_gpu=1 --n_embd=384 --n_ctx=93 --n_batch=64 --n_iter=10 --lr=1e-4 --n_head=12 --n_layer=6 --lm_coef=1.2 --data_dir=data/para.tsv --encoder_path=data/char_vocab.txt --desc=paraphrase --pos_weight=0.4 --new_model

