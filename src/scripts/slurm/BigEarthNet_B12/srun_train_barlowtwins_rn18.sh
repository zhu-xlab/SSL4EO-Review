#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_train_bt_rn18_crop_0_8_%j.out
#SBATCH --error=srun_outputs/B12_train_bt_rn18_crop_0_8_%j.err
#SBATCH --time=12:00:00
#SBATCH --job-name=bt_bigearthnet
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
srun python -u bigearthnet_B12_barlowtwins_train.py \
--data-path /p/project/hai_dm4eo/wang_yi/data/BigEarthNet \
--bands all \
--epochs 100 \
--batch-size 64 \
--workers 8 \
--lambd 0.0051 \
--projector 2048-2048-2048 \
--print-freq 100 \
--checkpoint-dir /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/barlowtwins/B12_rn18_2048_crop_0_8 \
--backbone resnet18 \
--dist_url $dist_url \
--schedule 60 80 \
--lr 0.2 \
--learning-rate-weights 0.1 \
--learning-rate-biases 0.0048 \
--weight-decay 1e-6 \
--cos \
