#!/usr/bin/env bash

export PYTHONPATH=.:$PYTHONPATH

#exp_name=segnet
#python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model segnet --loss default \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--test-data-file ./test_fc.pth \
#--pretrained True --with-bn True \
#--lr=0.01 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee logs/${exp_name}.log

#exp_name=segnet-no-bn
#python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model segnet --loss default \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--test-data-file ./test_fc.pth \
#--pretrained True \
#--thresh-epoch 20 \
#--lr=0.01 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee -a logs/${exp_name}.log

#exp_name=unet
#python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model unet --loss default \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--test-data-file ./test_fc.pth \
#--pretrained True --with-bn True \
#--thresh-epoch 20 --test-batch-size 3 \
#--lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee -a logs/${exp_name}.log

#exp_name=resnet50-fcn2s
#python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model fcn2s-resnet --loss default \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--test-data-file ./test_fc.pth \
#--pretrained True --with-bn True \d
#--thresh-epoch 20 --test-batch-size 3 --batch-size=16 \
#--lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee -a logs/${exp_name}.log

#exp_name=resnet152-fcn2s-sigmoid
#python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model fcn2s-resnet --type resnet152 --loss bce \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--test-data-file ./test_fc.pth \
#--pretrained True --with-bn True --none-classifier True --eval-method=sigmoid \
#--thresh-epoch 20 --test-batch-size 3 --batch-size=24 \
#--lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee -a logs/${exp_name}.log

#exp_name=vgg16-fcn2s
#python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model fcn2s --loss default \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
#--test-data-file ./test_fc.pth \
#--thresh-epoch 20 --test-batch-size 3 --batch-size=16 --pretrained=True \
#--lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee -a logs/${exp_name}.log

# TODO not run
# exp_name=vgg16-fcn2s-from-caffe
# python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model fcn2s --loss default \
# --train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair.txt \
# --test-data-file ./test_fc.pth \
# --thresh-epoch 20 --test-batch-size 3 --batch-size=16 --pretrained=True \
# --lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
# 2>&1 | tee -a logs/${exp_name}.log


#exp_name=resnet152-fcn2s-fc-st2
#python -u ./src/blur/train_test.py --exp-name ${exp_name} --model fcn2s-resnet-fc --with-fc True --type resnet152 --loss ce_fc \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair_fc.txt \
#--test-data-file ./test_fc.pth --test-data-with-fc=True --test-from-file=True \
#--pretrained True --with-bn True \
#--thresh-epoch 21 --test-batch-size 3 --batch-size=16 \
#--lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee -a logs/${exp_name}.log

exp_name=resnet152-fcn2s-fc-msrcnn2
python -u ./src/blur/train_test.py --exp-name ${exp_name} --model fcn2s-resnet-fc --eval-method msrcnn --with-fc True --type resnet152 --loss ms_fc \
--train-list-file ./dataset/list.txt \
--test-data-file ./test_fc.pth --test-data-with-fc=True --test-from-file=True \
--pretrained True --with-bn True \
--thresh-epoch 21 --test-batch-size 3 --batch-size=16 \
--lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
2>&1 | tee -a logs/${exp_name}.log


# fix bug： 初始训练fc的参数;另外增加了sample weight。
#exp_name=resnet152-fcn2s-fc-msrcnn-refine
#python -u ./src/pytorch/train_test.py --exp-name ${exp_name} --model fcn2s-resnet-fc --eval-method msrcnn --with-fc True --type resnet152 --loss ms_fc \
#--train-list-file /home/adam/Gits/blur-seg/grid_db/train_pair_fc.txt \
#--test-data-file ./test_fc.pth \
#--pretrained True --with-bn True \
#--thresh-epoch 21 --test-batch-size 3 --batch-size=16 \
#--lr=0.001 --weight-decay=0.00001 --n-epoch=100 --none-pretrained-first=True \
#2>&1 | tee -a logs/${exp_name}.log
## refine 的结果不是很好，这里并不知道什么原因，sample weight？可能的原因是motion和oof的都没训练好。