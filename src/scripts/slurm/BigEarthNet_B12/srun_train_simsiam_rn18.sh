#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_train_simsiam_rn18_crop_0_8_%j.out
#SBATCH --error=srun_outputs/B12_train_simsiam_rn18_crop_0_8_%j.err
#SBATCH --time=10:00:00
#SBATCH --job-name=simsiam_bigearthnet
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=booster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
module load GCCcore/.9.3.0
module load Python
module load torchvision
module load OpenCV
module load scikit
module load TensorFlow

# activate virtual environment
source /p/project/hai_dm4eo/wang_yi/env1/bin/activate

# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script as slurm job
srun python -u bigearthnet_B12_simsiam_train.py \
--data /p/project/hai_dm4eo/wang_yi/data/BigEarthNet \
--checkpoints /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/simsiam/B12_rn18_crop_0_8 \
--bands all \
--lmdb \
--arch resnet18 \
--workers 8 \
--batch-size 64 \
--epochs 100 \
--lr 0.05 \
--schedule 60 80 \
--momentum 0.9 \
--wd 1e-4 \
--dim 2048 \
--pred-dim 512 \
--print-freq 10 \
--dist-url $dist_url \
--dist-backend 'nccl' \
--seed 42 \
#--cos \
#--fix-pred-lr \
