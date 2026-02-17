import os
import argparse
from typing import *

import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from examples.trashnet_experiments.trashnet_class import TrashNetCustomClass
from examples.cifar10n_experiments.utils import read_json, write_json, seed_everything
from model import TrashNetClassifier

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--use_pretrain", action="store_true", default=False)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--run_name", type=str, default="new_validation")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=21094)
    parser.add_argument("--avoid_results_dump", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = command_line_args()
    seed_everything(args.seed)

    # Set up output paths
    if not args.avoid_results_dump:
        prefix = args.run_name
        default_path = 'experiments/out'
        output_path: str = f"{default_path}/{prefix}/{args.dataset_name}/"
        os.makedirs(output_path, exist_ok=True)

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load and split the dataset
    if args.dataset_name == 'TrashNet':
        dataset = TrashNetCustomClass(img_dir=os.getcwd() + '/data/dataset-resized', transform=transform)
    else:
        raise ValueError('Dataset not recognized!')

    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Initialize the model
    model = TrashNetClassifier(n_classes=6, pretrained=args.use_pretrain, aggregation_type=args.aggregation_type)

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        progress_bar_refresh_rate=20,
    )

    # Train the model
    trainer.fit(model, trainloader, valloader)

    # Test the model
    test_results = trainer.test(model, testloader)[0]

    # Save or print results
    if not args.avoid_results_dump:
        report_file_path = f"{output_path}/report_{args.seed}.json"
        if os.path.isfile(report_file_path):
            report = read_json(report_file_path)
        else:
            report = {}

        if args.aggregation_type not in report:
            report[args.aggregation_type] = {"pretrained": None, "scratch": None}
        report[args.aggregation_type]["pretrained" if args.use_pretrain else "scratch"] = test_results['test_acc']
        write_json(dictionary=report, filename=report_file_path)
    else:
        print(f"Test accuracy: {test_results['test_acc']}")