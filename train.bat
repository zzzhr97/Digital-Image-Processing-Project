python train.py ^
--task 1 ^
--seed 1 ^
--device cuda ^
--data_dir dataset ^
    ^
--n_valid 64 ^
--transform_method_origin 4 ^
--transform_method_epoch 2 ^
--batch_size 4 ^
--n_epochs 100 ^
--is_shuffle 1 ^
    ^
--optimizer adam ^
--lr 4e-5 ^
--lr_decay_epochs 30 60 80 ^
--lr_decay_values 2e-5 1e-5 5e-6 ^
--weight_decay 0.0001 ^
--betas 0.9 0.999 ^
--momentum 0.9 ^
    ^
--out_dim 2 ^
--threshold 0.5 ^
--model DenseNet ^
--is_search 0 ^
    ^
--ckpt_dir checkpoints ^
--result_dir results ^
--ckpt_every 100 ^
--eval_every 1 ^
--print_every 30 ^