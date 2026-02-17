#!/usr/bin/env bash

cd ../../

PYTHONPATH=. python3 examples/cifar10n_experiments/main.py --num_epochs 300 --run_name "test_cifar" --aggregation_type "posterior" --dataset_name "CIFAR-10N" --seed 21095 --use_pretrain &
PYTHONPATH=. python3 examples/cifar10n_experiments/main.py --num_epochs 300 --run_name "test_cifar" --aggregation_type "majority" --dataset_name "CIFAR-10N" --seed 21095 --use_pretrain &
PYTHONPATH=. python3 examples/cifar10n_experiments/main.py --num_epochs 300 --run_name "test_cifar" --aggregation_type "average" --dataset_name "CIFAR-10N" --seed 21095 --use_pretrain &
PYTHONPATH=. python3 examples/cifar10n_experiments/main.py --num_epochs 300 --run_name "test_cifar" --aggregation_type "random" --dataset_name "CIFAR-10N" --seed 21095 --use_pretrain &
PYTHONPATH=. python3 examples/cifar10n_experiments/main.py --num_epochs 300 --run_name "test_cifar" --aggregation_type "iwmv" --dataset_name "CIFAR-10N" --seed 21095 --use_pretrain &
PYTHONPATH=. python3 examples/cifar10n_experiments/main.py --num_epochs 300 --run_name "test_cifar" --aggregation_type "ds" --dataset_name "CIFAR-10N" --seed 21095 --use_pretrain &

