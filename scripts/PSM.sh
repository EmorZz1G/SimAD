#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/PSM/job.%j.out

python main.py --data_name PSM --win_size 2048 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 128 --step 1 --use_amp 0 --qlen
# python main.py --data_name PSM --win_size 4096 --lr 0.001 --gpu 0 --resume 0 --index 1 --batch_size 256 --step 1 --use_amp 0