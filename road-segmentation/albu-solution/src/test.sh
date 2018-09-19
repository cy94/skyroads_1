#!/usr/bin/env bash
set -e
arglen=$(($#-1))
args=${@:1:$arglen}
last=${!#}

echo "Args: $args"
echo "Last: $last"

rm -rf /wdata/*
python create_spacenet_masks.py $args
python train_eval.py resnet34_512_02_02.json
python merge_preds.py
python skeleton.py $last
