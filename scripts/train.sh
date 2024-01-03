#!/bin/bash

python src/train.py \
--task 1 \
--seed 1 \
--device cuda \
--data_dir dataset \
\
--k_fold 0 \
--n_valid 64 \
--transform_method_origin 1 \
--transform_method_epoch 2 \
--batch_size 4 \
--n_epochs 5 \
--is_shuffle 1 \
\
--optimizer adam \
--lr 0.0001 \
--lr_decay_epochs 15 \
--lr_decay_values 0.00001 \
--weight_decay 0.0001 \
--betas 0.9 0.999 \
--momentum 0.9 \
\
--out_dim 1 \
--threshold 0.5 \
--model TestNet \
--is_search 0 \
\
--ckpt_dir checkpoints \
--result_dir results \
--ckpt_every 10 \
--eval_every 1 \
--print_every 2 \
\
--pretrained 0 \
--ckpt_load_dir checkpoints_load \
--ckpt_load_name resnet_pretrain \