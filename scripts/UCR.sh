#!/bin/sh
#SBATCH -p gpuA800 
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1 
#SBATCH --output ./logs/UCR/job.%j.out

# python main.py --data_name UCR --win_size 2048 --lr 0.01 --gpu 0 --resume 0 --index 1 --batch_size 256 --step 1
# for i in {1..250}
# do 
# python main.py --data_name UCR --win_size 2048 --lr 0.01 --gpu 0 --resume 0 --index $i --batch_size 256 --step 1
# done

# subset
# s1_data=(26 27 28 32 34 53 68 98 99 134 135 136 140 142 161 176 248)
# s1_data=(33 36 54 79 108 141 144 162 187 203 212 229)
# s1_data=(79)

# for i in {1..250}; do
# # for i in ${s1_data[@]}
# # do 
# echo $i
# python main.py --data_name UCR --win_size 2048 --lr 0.001 --gpu -1 --resume 1 --index $i --batch_size 128 --step 1 --epochs 30 --patch_size 32
# done


# for i in {1..250}; do
for i in {210..250}; do
# for i in ${s1_data[@]}
# do 
echo $i
python main.py --data_name UCR --win_size 2048 --lr 0.001 --gpu -1 --resume 1 --index $i --batch_size 128 --step 1 --epochs 30 --patch_size 16
done

# for i in {1..250}; do
# s1_data=(26 27 28 32 34 53 68 98 99 134 135 136 140 142 161 176 248)
# for i in ${s1_data[@]}
# do 
# echo $i
# python main.py --data_name UCR --win_size 128 --lr 0.001 --gpu -1 --resume 1 --index $i --batch_size 128 --step 1 --epochs 30 --patch_size 16
# done

# s1_data=(33 36 54 79 108 141 144 162 187 203 212 229)
# for i in ${s1_data[@]}
# do 
# echo $i
# python main.py --data_name UCR --win_size 128 --lr 0.001 --gpu -1 --resume 1 --index $i --batch_size 128 --step 1 --epochs 30 --patch_size 8
# done
# i=82
# python main.py --data_name UCR --win_size 512 --lr 0.0001 --gpu 0 --resume 0 --index $i --batch_size 128 --step 1 --epochs 35 --warmup_max_ratio 0.3 --warmup_steps 1000