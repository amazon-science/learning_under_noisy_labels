cd ../..


PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type posterior --dataset_language sp --num_epochs 100 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type ds --dataset_language sp --num_epochs 100 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type majority --dataset_language sp --num_epochs 100 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type iwmv --dataset_language sp --num_epochs 100 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type average --dataset_language sp --num_epochs 100 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type random --dataset_language sp --num_epochs 100 --run_name experiment_textual --seed 21096

PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type posterior --dataset_language en --num_epochs 300 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type ds --dataset_language en --num_epochs 300 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type majority --dataset_language en --num_epochs 300 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type iwmv --dataset_language en --num_epochs 300 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type average --dataset_language en --num_epochs 300 --run_name experiment_textual --seed 21096
PYTHONPATH=. python3 examples/text_experiments/text_classification.py --aggregation_type random --dataset_language en --num_epochs 300 --run_name experiment_textual --seed 21096