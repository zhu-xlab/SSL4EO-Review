#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_ft_moco_rn18_10_%j.out
#SBATCH --error=srun_outputs/B12_ft_moco_rn18_10_%j.err
#SBATCH --time=01:00:00
#SBATCH --job-name=moco_ft
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=develbooster

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
srun python -u bigearthnet_B12_moco_FT.py \
--lmdb_dir /p/project/hai_dm4eo/wang_yi/data/BigEarthNet \
--bands all \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/moco_ft/B12_FT_moco_rn18_10 \
--save_path /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/moco_ft/B12_FT_moco_rn18_10.pth.tar \
--backbone resnet18 \
--train_frac 0.1 \
--batchsize 256 \
--lr 0.1 \
--lr_last_layer 1.0 \
--schedule 20 30 \
--epochs 50 \
--num_workers 8 \
--seed 42 \
--dist_url $dist_url \
--pretrained /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/moco/B12_rn18/checkpoint_0199.pth.tar \
