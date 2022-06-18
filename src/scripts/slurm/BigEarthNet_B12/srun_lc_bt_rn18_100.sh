#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_lc_bt_rn18_100_crop_0_8_%j.out
#SBATCH --error=srun_outputs/B12_lc_bt_rn18_100_crop_0_8_%j.err
#SBATCH --time=05:00:00
#SBATCH --job-name=barlowtwins_lc
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
srun python -u bigearthnet_B12_barlowtwins_LC.py \
--lmdb_dir /p/project/hai_dm4eo/wang_yi/data/BigEarthNet \
--bands all \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/barlowtwins_lc/B12_rn18_100_crop_0_8 \
--save_path /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/barlowtwins_lc/B12_rn18_100_crop_0_8.pth.tar \
--backbone resnet18 \
--train_frac 1.0 \
--batchsize 256 \
--lr 4 \
--schedule 60 80 \
--epochs 100 \
--num_workers 8 \
--seed 42 \
--dist_url $dist_url \
--pretrained /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/barlowtwins/B12_rn18_2048_crop_0_8/checkpoint.pth \
