#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B10_lc_simsiam_rn18_100_%j.out
#SBATCH --error=srun_outputs/B10_lc_simsiam_rn18_100_%j.err
#SBATCH --time=04:00:00
#SBATCH --job-name=simsiam_lc
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
srun python -u so2sat_B10_simsiam_LC.py \
--data_dir /p/project/hai_dm4eo/wang_yi/data/so2sat-lcz42 \
--bands B10 \
--checkpoints_dir /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/so2sat/simsiam_lc/so2sat_B10_LC_simsiam_rn18_100 \
--save_path /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/so2sat/simsiam_lc/so2sat_B10_LC_simsiam_rn18_100.pth.tar \
--backbone resnet18 \
--train_frac 1.0 \
--batchsize 256 \
--lr 2 \
--schedule 20 30 \
--epochs 40 \
--num_workers 8 \
--seed 42 \
--pretrained /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/so2sat/simsiam/B10_rn18/checkpoint_0009.pth.tar \
--dist_url $dist_url \
