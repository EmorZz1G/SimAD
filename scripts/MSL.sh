#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/MSL/job.%j.out

python main.py --data_name MSL --win_size 25600 --lr 0.0001 --gpu -1 --resume 0 --index 1 --batch_size 32 --step 1 --use_amp 0 --warmup_steps 500 --warmup_max_ratio 0.3
# python main.py --data_name MSL --win_size 25600 --lr 0.0001 --gpu -1 --resume 1 --index 1 --batch_size 32 --step 1 --use_amp 0 --warmup_steps 500 --warmup_max_ratio 0.3 --resume 1