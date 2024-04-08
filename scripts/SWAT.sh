#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/SWAT/job.%j.out

# python main.py --data_name SWAT --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 256 --step 1 --use_amp 0 --warmup_steps 20000 --warmup_max_ratio 0.2
# python main.py --data_name SWAT --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 128 --step 1 --use_amp 0 --warmup_steps 20000 --warmup_max_ratio 0.2 
# python main.py --data_name SWAT --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 128 --step 1 --use_amp 0 --warmup_steps 1000 --warmup_max_ratio 0.3
# python main.py --data_name SWAT --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 128 --step 1 --use_amp 0 --warmup_steps 1000 --warmup_max_ratio 0.3 --patch_size 16
# python main.py --data_name SWAT --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 64 --step 20 --use_amp 0 --warmup_steps 1000 --warmup_max_ratio 0.3 --patch_size 32


python main.py --data_name SWAT --win_size 2048 --lr 0.001 --gpu 0 --resume 1 --index 1 --batch_size 256 --step 1 --use_amp 1 --warmup_steps 20000 --warmup_max_ratio 0.2 --model_id 2024_03_11_10:59:56