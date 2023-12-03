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
--n_epochs 2 ^
--lr 0.0001 ^
--lr_decay_epochs 15 30 50 ^
--lr_decay_values 0.0001 0.00002 0.00001 ^
--weight_decay 0.0001 ^
--is_shuffle 1 ^
--optimizer adam ^
    ^
--out_dim 1 ^
--threshold 0.5 ^
--model TestNet ^
--is_search 0 ^
    ^
--ckpt_dir checkpoints ^
--result_dir results ^
--ckpt_every 100 ^
--eval_every 1 ^
--print_every 100 ^