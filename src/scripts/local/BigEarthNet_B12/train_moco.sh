python moco_bigearthnet_B12_train.py \
--data /mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb \
--checkpoints /mnt/d/codes/ssl4eo-review/src/checkpoints/moco/checkpoints \
--save_path /mnt/d/codes/ssl4eo-review/src/checkpoints/moco/ \
--bands all \
--lmdb \
--arch resnet50 \
--workers 8 \
--batch-size 256 \
--epochs 200 \
--lr 0.03 \
--mlp \
--moco-t 0.2 \
--aug-plus \
--cos \
--seed 42 \
--gpu 0 \
#--resume '' \
#--fix-pred-lr \
#--dist-url 'tcp://224.66.41.62:23456' \
#--dist-backend 'nccl' \
#--world-size 1 \
#--rank 0 \
#--multiprocessing-distributed \

