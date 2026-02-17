#!/usr/bin/env bash

cd ../../


PYTHONPATH=. python3 examples/trashnet_experiments/trashnet_train.py --run_name "trashnet_test" --aggregation_type "average"  --dataset_name "TrashNet" --num_epochs 300 --use_pretrain 
PYTHONPATH=. python3 examples/trashnet_experiments/trashnet_train.py --run_name "trashnet_test" --aggregation_type "majority"  --dataset_name "TrashNet" --num_epochs 300 --use_pretrain 
PYTHONPATH=. python3 examples/trashnet_experiments/trashnet_train.py --run_name "trashnet_test" --aggregation_type "random"  --dataset_name "TrashNet" --num_epochs 300 --use_pretrain 
PYTHONPATH=. python3 examples/trashnet_experiments/trashnet_train.py --run_name "trashnet_test" --aggregation_type "iwmv"  --dataset_name "TrashNet" --num_epochs 300 --use_pretrain 
PYTHONPATH=. python3 examples/trashnet_experiments/trashnet_train.py --run_name "trashnet_test" --aggregation_type "ds"  --dataset_name "TrashNet" --num_epochs 300 --use_pretrain
PYTHONPATH=. python3 examples/trashnet_experiments/trashnet_train.py --run_name "trashnet_test" --aggregation_type "posterior"  --dataset_name "TrashNet" --num_epochs 300 --use_pretrain 