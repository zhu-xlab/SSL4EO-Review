#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/B12_train_swav_rn50s_%j.out
#SBATCH --error=srun_outputs/B12_train_swav_rn50s_%j.err
#SBATCH --time=10:00:00
#SBATCH --job-name=swav_bigearthnet
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
srun python -u bigearthnet_B12_swav_train.py \
--data_path /p/project/hai_dm4eo/wang_yi/data/BigEarthNet \
--dump_path /p/project/hai_dm4eo/wang_yi/ssl4eo-review/src/checkpoints/swav/B12_rn50_s \
--lmdb \
--bands all \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 60 \
--queue_length 0 \
--epochs 100 \
--batch_size 64 \
--workers 8 \
--base_lr 0.2 \
--final_lr 0.0005 \
--cos \
--freeze_prototypes_niters 313 \
--wd 0.0001 \
--arch resnet50 \
--use_fp16 false \
--sync_bn pytorch \
--dist_url $dist_url \
#--nmb_crops 2 6 \
#--size_crops 224 96 \
#--min_scale_crops 0.14 0.05 \
#--max_scale_crops 1. 0.14 \
#--crops_for_assign 0 1 \
#--warmup_epochs 10 \
#--start_warmup 0.3 \
#--schedule 60 80 \
