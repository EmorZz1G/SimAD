#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/UCR/job.%j.out

for i in {1..250}; do
do 
echo $i
python main.py --data_name UCR --win_size 256 --lr 0.001 --gpu -1 --resume 1 --index $i --batch_size 128 --step 1 --epochs 30 --patch_size 8
done
