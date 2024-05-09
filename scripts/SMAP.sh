#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/SMAP/job.%j.out

# python main.py --data_name SMAP --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 128 --step 1 --use_amp 0 --epochs 30 --patch_size 32 --warmup_max_ratio 0.2 --warmup_steps 10000 --d_model 512 --noise_level 0.1
python main.py --data_name SMAP --win_size 128 --lr 0.01 --gpu 0 --resume 0 --index 1 --batch_size 512 --step 1 --use_amp 1 --epochs 100 --patch_size 4 --warmup_max_ratio 0.1 --warmup_steps 20000 --d_model 128 --noise_level 0.1 --q_len 8 --test_freq 500