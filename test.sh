#!/usr/bin/env bash

export PYTHONPATH=.:$PYTHONPATH

#exp_name="fcn2s-resnet101-k3-weighted-loss"
#check_point_num=70
#python -u test.py test --check-point models/${exp_name}/epoch-0${check_point_num}.pth \
#--rst-dir results/${exp_name} \
#--test-list-file /home/adam/Gits/blur-seg/grid_db/val_pair.txt \
#--models fcn2s-resnet \
#--type resnet101

#exp_name="fcn2s-resnet152-k3-fc"
#check_point_num=50
#python -u test.py test --check-point models/${exp_name}/epoch-0${check_point_num}.pth \
#--rst-dir results/${exp_name} \
#--test-list-file /home/adam/Gits/blur-seg/grid_db/val_pair_fc.txt \
#--models fcn2s-resnet-fc \
#--type resnet152

#exp_name="fcn2s-resnet152-k3-fc-msrcnn"
#check_point_num=40
#python -u test.py test --check-point model/${exp_name}/epoch-0${check_point_num}.pth \
#--rst-dir results/${exp_name} \
#--test-list-file /home/adam/Gits/blur-seg/grid_db/val_pair_fc.txt \
#--model fcn2s-resnet-fc \
#--type resnet152 \
#--eval-method msrcnn

exp_name=resnet152-fcn2s-sigmoid
check_point_num=80
python -u src/blur/test.py test --check-point models/${exp_name}/epoch-0${check_point_num}.pth \
--rst-dir results/${exp_name} \
--test-list-file /home/adam/Gits/blur-seg/grid_db/val_raw_pair.txt \
--model fcn2s-resnet \
--type resnet152 \
--eval-method sigmoid

#exp_name="fcn2s-resnet152-k3"
#check_point_num=50
#python -u test.py test --check-point models/${exp_name}/epoch-0${check_point_num}.pth \
#--rst-dir results/${exp_name} \
#--test-list-file /home/adam/Gits/blur-seg/grid_db/val_pair.txt \
#--models fcn2s-resnet \
#--type resnet152


# FIXME 模型不同
#exp_name="fcn2s-resnet50"
#check_point_num=50
#python -u test.py test --check-point models/${exp_name}/epoch-0${check_point_num}.pth \
#--rst-dir results/${exp_name} \
#--test-list-file /home/adam/Gits/blur-seg/grid_db/val_pair.txt \
#--models fcn2s-resnet \
#--type resnet50