python train.py ^
--task 1 ^
--seed 1 ^
--device cuda ^
--data_dir dataset ^
    ^
--n_valid 64 ^
--transform_method_origin 1 ^
--transform_method_epoch 2 ^
--batch_size 4 ^
--n_epochs 100 ^
--is_shuffle 1 ^
    ^
--optimizer adam ^
--lr 0.0002 ^
--lr_decay_epochs 50 80 901 ^
--lr_decay_values 0.0001 0.00004 0.0001 ^
--weight_decay 0.0001 ^
--betas 0.9 0.999 ^
--momentum 0.9 ^
    ^
--out_dim 2 ^
--threshold 0.5 ^
--model SResNet ^
--is_search 0 ^
    ^
--ckpt_dir checkpoints ^
--result_dir results ^
--ckpt_every 100 ^
--eval_every 1 ^
--print_every 30 ^