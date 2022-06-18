#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B10_lc_sup_rn18_10_%j.out
#SBATCH --error=srun_outputs/B10_lc_sup_rn18_10_%j.err
#SBATCH --time=04:00:00
#SBATCH --job-name=sup_lc
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
srun python -u so2sat_B10_resnet_LC.py \
--data_dir /p/project/hai_dm4eo/wang_yi/data/so2sat-lcz42/ \
--bands B10 \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/so2sat/sup_lc/B10_rn18_10 \
--backbone resnet18 \
--train_frac 0.1 \
--batchsize 256 \
--lr 0.2 \
--schedule 20 40 \
--epochs 50 \
--num_workers 8 \
--seed 42 \
--dist_url $dist_url \
