#!/usr/bin/env bash
# python ./src/pytorch/train.py  --train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt --batch-size=10 --lr=0.01

# train resnet
#exp_name="fcn2s-resnet50-k3"
#python -u /home/adam/Gits/blur-detection/src/pytorch/train.py --models fcn2s-resnet \
#--type=resnet50 \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--batch-size=10 --lr=0.001 \
#--models-root=models/${exp_name} \
#2>&1 | tee logs/${exp_name}.log

#exp_name="fcn2s-resnet152-k3"
#python -u /home/adam/Gits/blur-detection/src/pytorch/train.py --models fcn2s-resnet \
#--type=resnet152 \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--batch-size=10 --lr=0.001 \
#--models-root=models/${exp_name} \
#2>&1 | tee logs/${exp_name}.log

#exp_name="fcn2s-resnet152-k3-fc"
#python -u /home/adam/Gits/blur-detection/src/pytorch/train.py \
#--models fcn2s-resnet-fc \
#--type=resnet152 \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair_fc.txt \
#--batch-size=10 --lr=0.001 \
#--models-root=models/${exp_name} \
#2>&1 | tee logs/${exp_name}.log

#exp_name="fcn2s-resnet152-k3-fc-msrcnn"
#python -u /home/adam/Gits/blur-detection/src/pytorch/train.py \
#--models fcn2s-resnet-fc \
#--type=resnet152 \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair_fc.txt \
#--batch-size=15 --lr=0.001 \
#--loss=bceloss \
#--epoch=50 \
#--exp=${exp_name} \
#2>&1 | tee -a logs/${exp_name}.log
#--resume=True \
#--check-point=models/${exp_name}/epoch-050.pth \
#--models-root=models/${exp_name} \

# 2017-12-11 22:33
#exp_name="fcn2s-resnet152-k3-fc-hsv"
#python -u /home/adam/Gits/blur-detection/src/pytorch/train.py \
#--model fcn2s-resnet-fc \
#--type=resnet152 \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair_fc.txt \
#--epoch=70 \
#--loss=fc \
#--batch-size=28 --lr=0.01 \
#--exp=${exp_name} \
#2>&1 | tee -a logs/${exp_name}.log
#--batch-size=15 --lr=0.001 \

## 从第50次开始训练
#exp_name="fcn2s-resnet101-k3-weighted-loss"
#python -u /home/adam/Gits/blur-detection/src/pytorch/train.py --models fcn2s-resnet \
#--type=resnet101 \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--batch-size=10 --lr=0.001 \
#--models-root=models/${exp_name} \
#--resume=True \
#--epoch=70 \
#--check-point=models/${exp_name}/epoch-050.pth \
#2>&1 | tee -a logs/${exp_name}.log


# train densenet121
#python -u /home/adam/Gits/blur-detection/src/pytorch/train.py --models fcn8s-densenet \
#--type=121 \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--batch-size=10 --lr=0.001 \
#--mode-root=models/fcn8s-densenet121
#2>&1 | tee logs/fcn8s-densenet121.log


exp_name="fcn2s-resnet152-k3-fc-hsv"
python -u ./src/blur/train.py \
--model fcn2s-resnet-fc \
--type=resnet152 \
--train-list-file ./dataset/list.txt \
--epoch=50 \
--loss=fc \
--batch-size=28 --lr=0.01 \
--exp=${exp_name} \
2>&1 | tee -a logs/${exp_name}.log