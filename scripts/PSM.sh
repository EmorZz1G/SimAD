#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/PSM/job.%j.out

python main.py --data_name PSM --win_size 2048 --lr 0.001 --gpu -1 --resume 0 --index 1 --batch_size 128 --step 1 --use_amp 0 --warmup_steps 20000 --warmup_max_ratio 0.2