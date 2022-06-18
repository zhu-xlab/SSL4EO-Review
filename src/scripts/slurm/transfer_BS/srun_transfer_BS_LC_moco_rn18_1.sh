#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/BS_lc_moco_rn18_1_%j.out
#SBATCH --error=srun_outputs/BS_lc_moco_rn18_1_%j.err
#SBATCH --time=00:50:00
#SBATCH --job-name=BS_moco_lc
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
srun python -u transfer_BS_moco_LC.py \
--data_dir /p/project/hai_dm4eo/wang_yi/data/so2sat-lcz42 \
--bands B12 \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/transfer_BS/moco_lc/BS_LC_moco_rn18_1 \
--save_path /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/transfer_BS/moco_lc/BS_LC_moco_rn18_1.pth.tar \
--backbone resnet18 \
--train_frac 0.01 \
--batchsize 64 \
--lr 0.5 \
--schedule 40 60 \
--epochs 100 \
--num_workers 8 \
--seed 42 \
--dist_url $dist_url \
--pretrained /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/moco/B12_rn18/checkpoint_0199.pth.tar
