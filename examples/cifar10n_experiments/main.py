import os
import argparse
from typing import *

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from examples.cifar10n_experiments.cifar10_class import CIFAR10CustomClass
from examples.cifar10n_experiments.utils import *
from model import CIFAR10Model


def main(args):
    """
    This script is used to train and evaluate a ResNet34 model on the CIFAR-10 dataset with noisy annotations.

    The main function `main` takes the following command-line arguments:
        - aggregation_type: The type of aggregation method to use for noisy annotations.
        - batch_size: The batch size for training and evaluation.
        - patient: Not used in the provided code.
        - num_annotators: Not used in the provided code.
        - noise_percentage: Not used in the provided code.
        - use_pretrain: Whether to use a pretrained ResNet34 model or train from scratch.
        - dataset_name: Not used in the provided code.
        - run_name: The name of the run, used for creating output directories.
        - num_epochs: The number of epochs to train the model.
        - seed: The random seed for reproducibility.

    The script performs the following steps:
    1. Set up output directories based on the run_name and dataset_name.
    2. Load the CIFAR-10 dataset and split it into train, validation, and test sets.
    3. Load noisy annotations for the training set using the specified aggregation_type.
    4. Create custom CIFAR-10 datasets and data loaders.
    5. Initialize the CIFAR10Model with the specified pretrained option.
    6. Train the model using PyTorch Lightning Trainer.
    7. Test the model on the test set and calculate the test accuracy.
    8. Save the test accuracy in a JSON report file.

    The script also includes helper functions:
    - get_percentage: Calculates a percentage of a total value.
    - get_train_val_split: Splits the dataset into train and validation sets.
    - load_cifar10n: Loads noisy annotations for the CIFAR-10 dataset.

    The CIFAR10Model class is a PyTorch Lightning module that defines the ResNet34 model, loss function,
    training step, validation step, test step, and optimizer.
    """

    # Set random seed for reproducibility
    seed_everything(seed=args.seed)

    # Set up output directories
    prefix = args.run_name
    default_path = "experiments/out"
    output_path: str = f"{default_path}/{prefix}/{args.dataset_name}/"

    os.makedirs(f"{default_path}/{prefix}/{args.dataset_name}", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Define data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load CIFAR10 dataset
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Load noisy annotations
    dataset.targets = load_cifar10n(args.aggregation_type, dataset)
    train_set, train_data, val_set, val_data, train_set_indices, val_set_indices = (
        get_train_val_split(dataset, seed=args.seed)
    )

    # Create custom CIFAR10 datasets
    train_set = CIFAR10CustomClass(
        train_set.data, train_set.targets, test_set.transform
    )
    val_set = CIFAR10CustomClass(val_set.data, val_set.targets, test_set.transform)

    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Initialize the model
    model = CIFAR10Model(n_classes=10, pretrained=args.use_pretrain)

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus=1 if torch.cuda.is_available() else None,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Save results
    test_results = trainer.test(model, test_loader)[0]
    report_file_path = f"{output_path}/report_{args.seed}.json"

    # Update and save the report
    if os.path.isfile(report_file_path):
        report = read_json(report_file_path)
    else:
        report = {}

    if args.aggregation_type not in report:
        report[args.aggregation_type] = {"pretrained": None, "scratch": None}
    report[args.aggregation_type]["pretrained" if args.use_pretrain else "scratch"] = (
        test_results["test_acc"]
    )
    write_json(dictionary=report, filename=report_file_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patient", type=int, default=100)
    parser.add_argument("--num_annotators", type=int, default=4)
    parser.add_argument("--noise_percentage", type=float, default=0.40)
    parser.add_argument("--use_pretrain", action="store_true", default=False)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--run_name", type=str, default="new_validation")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=21094)
    args = parser.parse_args()

    main(args)
