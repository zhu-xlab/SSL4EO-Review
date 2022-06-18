python -m torch.distributed.launch --nproc_per_node=1 bigearthnet_B12_resnet_train.py \
--data_dir /mnt/d/codes/SSL_examples/datasets/BigEarthNet \
--lmdb_dir /mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb \
--checkpoints_dir /mnt/d/codes/ssl4eo-review/src/checkpoints/resnet/B12_rn18 \
--save_path /mnt/d/codes/ssl4eo-review/src/checkpoints/resnet_lc/bigearthnet_B12_train_resnet_rn50_100.pth.tar \
--backbone resnet50 \
--bands all \
--train_frac 1.0 \
--batchsize 256 \
--lr 0.05 \
--schedule 60 80 \
--epochs 100 \
--num_workers 8 \
--seed 42 \