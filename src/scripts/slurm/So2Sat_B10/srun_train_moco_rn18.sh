#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B10_train_moco_rn18_%j.out
#SBATCH --error=srun_outputs/B10_train_moco_rn18_%j.err
#SBATCH --time=05:00:00
#SBATCH --job-name=moco_so2sat
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=booster

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
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
srun python -u so2sat_B10_moco_train.py \
--data /p/project/hai_dm4eo/wang_yi/data/so2sat-lcz42 \
--checkpoints /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/so2sat/moco/B10_rn18 \
--bands B10 \
--arch resnet18 \
--workers 8 \
--batch-size 64 \
--epochs 100 \
--lr 0.05 \
--mlp \
--moco-t 0.2 \
--aug-plus \
--schedule 60 80 \
--dist-url $dist_url \
--dist-backend 'nccl' \
--seed 42 \
