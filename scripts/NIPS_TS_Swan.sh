#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/NIPS_TS_Swan/job.%j.out



python main.py --data_name NIPS_TS_Swan --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 128 --step 1 --use_amp 1 --warmup_max_ratio 0.2 --warmup_steps 10000 --epochs 40 --patch_size 32