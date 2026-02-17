from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import argparse
import os

from examples.cifar10n_experiments.utils import read_json, write_json, seed_everything
from text_class import PersTecData
from model import TextClassifier


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregation_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset_language", type=str, default="en")
    parser.add_argument("--run_name", type=str, default="new_validation")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=21094)
    parser.add_argument("--avoid_results_dump", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = command_line_args()
    seed_everything(args.seed)
    if not args.avoid_results_dump:
        default_path = 'experiments/out'
        output_path: str = f"{default_path}/{args.run_name}/{args.dataset_language}"
        report_file_path = f"{output_path}/report_{args.seed}.json"

    dataset = PersTecData(language=args.dataset_language, aggregation_type=args.aggregation_type)

    remaining = len(dataset) - (int(0.2 * len(dataset)) + int(0.7 * len(dataset)) + int(0.1 * len(dataset)))
    train_dataset, val_dataset, test_dataset = random_split(dataset, (int(0.7 * len(dataset)) + remaining,
                                                                      int(0.2 * len(dataset)),
                                                                      int(0.1 * len(dataset))))

    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)

    num_labels = max(dataset.y) + 1
    model = TextClassifier(num_labels=num_labels, language=args.dataset_language)

    # Train the model
    trainer = pl.Trainer(max_epochs=args.num_epochs , gpus=1,
                         checkpoint_callback=False, logger=False)
    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(model, test_loader)
    if not args.avoid_results_dump:
        if os.path.isfile(report_file_path):
            report = read_json(report_file_path)
        else:
            report = {}
            os.makedirs(f"{default_path}/{args.run_name}/{args.dataset_language}", exist_ok=True)
            os.makedirs(output_path, exist_ok=True)
            write_json(f"{output_path}/report_{args.seed}.json", report)

        if args.aggregation_type not in report:
            report[args.aggregation_type] = {}
            for metric, res in results[0].items():
                report[args.aggregation_type][metric] = res

        write_json(dictionary=report, filename=f"{output_path}/report_{args.seed}.json")
    else:
        print(results)
