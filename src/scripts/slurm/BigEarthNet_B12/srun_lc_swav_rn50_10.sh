#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_lc_swav_rn50_10_%j.out
#SBATCH --error=srun_outputs/B12_lc_swav_rn50_10_%j.err
#SBATCH --time=01:00:00
#SBATCH --job-name=swav_lc
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
srun python bigearthnet_B12_swav_LC.py \
--lmdb_dir /p/project/hai_dm4eo/wang_yi/data/BigEarthNet \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/swav_lc/B12_LC_swav_rn50_10 \
--save_path /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/swav_lc/B12_LC_swav_rn50_10.pth.tar \
--backbone resnet50 \
--bands all \
--train_frac 0.1 \
--batchsize 64 \
--lr 1.0 \
--schedule 60 80 \
--epochs 100 \
--num_workers 8 \
--seed 42 \
--pretrained /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/swav/crop_only/B12_rn50_m/checkpoint.pth.tar \
--dist_url $dist_url \
