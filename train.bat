python train.py ^
--task 1 ^
--seed 1 ^
--device cuda ^
--data_dir dataset ^
    ^
--k_fold 5 ^
--n_valid 128 ^
--transform_method_origin 1 ^
--transform_method_epoch 2 ^
--batch_size 4 ^
--n_epochs 2 ^
--is_shuffle 1 ^
    ^
--optimizer adam ^
--lr 1e-4 ^
--lr_decay_epochs 30 60 90 120 ^
--lr_decay_values 5e-5 2e-5 1e-5 5e-6 ^
--weight_decay 0.0016 ^
--betas 0.9 0.999 ^
--momentum 0.9 ^
    ^
--out_dim 2 ^
--threshold 0.5 ^
--model TestNet ^
--is_search 0 ^
    ^
--ckpt_dir checkpoints ^
--result_dir results ^
--ckpt_every 100 ^
--eval_every 1 ^
--print_every 30 ^
    ^
--pretrained 1 ^
--ckpt_load_dir checkpoints_load ^
--ckpt_load_name pre ^