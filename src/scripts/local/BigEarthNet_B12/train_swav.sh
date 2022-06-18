python -m torch.distributed.launch --nproc_per_node=1 swav_bigearthnet_B12_train.py \
--data_path /mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb \
--dump_path /mnt/d/codes/ssl4eo-review/src/checkpoints/swav/checkpoints \
--lmdb \
--bands all \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 0 \
--epochs 100 \
--batch_size 64 \
--workers 2 \
--base_lr 0.3 \
--final_lr 0.0048 \
--freeze_prototypes_niters 313 \
--wd 0.0001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet50 \
--use_fp16 false \
--sync_bn pytorch \
#--nmb_crops 2 6 \
#--size_crops 224 96 \
#--min_scale_crops 0.14 0.05 \
#--max_scale_crops 1. 0.14 \
#--crops_for_assign 0 1 \
#--dist_url $dist_url \

