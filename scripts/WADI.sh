#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/WADI/job.%j.out



python main.py --data_name WADI --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 256 --step 1 --use_amp 1 --warmup_steps 20000 --warmup_max_ratio 0.2 --epochs 15
