python simsiam_bigearthnet_B12_train.py \
--data /mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb \
--checkpoints /mnt/d/codes/ssl4eo-review/src/checkpoints/simsiam/checkpoints \
--save_path /mnt/d/codes/ssl4eo-review/src/checkpoints/simsiam/ \
--lmdb \
--bands all \
--arch resnet50 \
--workers 8 \
--gpu 0 \
--batch-size 256 \
--epochs 100 \
--lr 0.05 \
--momentum 0.9 \
--wd 1e-4 \
--dim 2048 \
--pred-dim 512 \
--print-freq 10 \
--seed 42 \
#--resume '' \
#--fix-pred-lr \
#--dist-url 'tcp://224.66.41.62:23456' \
#--dist-backend 'nccl' \
#--world-size 1 \
#--rank 0 \
#--multiprocessing-distributed \

