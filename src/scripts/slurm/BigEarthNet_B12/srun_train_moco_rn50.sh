#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_train_moco_rn50_%j.out
#SBATCH --error=srun_outputs/B12_train_moco_rn50_%j.err
#SBATCH --time=10:00:00
#SBATCH --job-name=moco_bigearthnet
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
srun python -u bigearthnet_B12_moco_train.py \
--data /p/project/hai_dm4eo/wang_yi/data/BigEarthNet \
--checkpoints /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/moco/B12_rn50_400ep \
--bands all \
--lmdb \
--arch resnet50 \
--workers 8 \
--batch-size 256 \
--epochs 400 \
--lr 0.2 \
--mlp \
--moco-t 0.2 \
--aug-plus \
--schedule 120 160 \
--cos \
--dist-url $dist_url \
--dist-backend 'nccl' \
--seed 42 \
