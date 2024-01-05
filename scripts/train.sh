#!/bin/bash

python src/train.py \
--task 1 \
--seed 135 \
--device cuda \
--data_dir dataset \
\
--k_fold 5 \
--n_valid 128 \
--transform_method_origin 1 \
--transform_method_epoch 2 \
--batch_size 4 \
--n_epochs 5 \
--is_shuffle 1 \
\
--optimizer adam \
--lr 0.0001 \
--lr_decay_epochs 2 4 \
--lr_decay_values 0.0005 0.0002 \
--weight_decay 0.0001 \
--betas 0.9 0.999 \
--momentum 0.9 \
\
--out_dim 2 \
--threshold 0.5 \
--model ResNet18 \
--is_search 0 \
\
--ckpt_dir checkpoints \
--result_dir results \
--ckpt_every 1 \
--eval_every 1 \
--print_every 30 \
\
--pretrained 1 \
--ckpt_load_dir checkpoints_load \
--ckpt_load_name resnet_pretrain \