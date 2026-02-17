#!/usr/bin/env bash


cd ../../

PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="dawidskene" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=42 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="dawidskene" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=43 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="dawidskene" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=44 data.equal_distribution=False data.noise_matrix_type=symmetric &

PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="iwmv" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=42 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="iwmv" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=43 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="iwmv" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=44 data.equal_distribution=False data.noise_matrix_type=symmetric &

PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="average" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=42 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="average" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=43 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="average" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=44 data.equal_distribution=False data.noise_matrix_type=symmetric &

PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="posterior" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=42 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="posterior" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=43 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="posterior" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=44 data.equal_distribution=False data.noise_matrix_type=symmetric &

PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="majority" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=42 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="majority" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=43 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="majority" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=44 data.equal_distribution=False data.noise_matrix_type=symmetric &


PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="random" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=42 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="random" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=43 data.equal_distribution=False data.noise_matrix_type=symmetric &
PYTHONPATH=. python3 examples/syntethic_experiments/src/train.py -m data.aggregation_type="random" data.min_t_diag_value=0.6,0.7,0.8,0.9 train.experiment_name=experiment_zero data.num_annotators=3 train.seed=44 data.equal_distribution=False data.noise_matrix_type=symmetric &

echo "done"
