import json
import os
from collections import defaultdict
import json
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import argparse
import statistics
from collections import OrderedDict

from examples.syntethic_experiments.src.utils import read_json


def create_results_table(cluster, metrics, methods):
    """
    Given the results of the synthetic experiments, this method aggregates them based on the minimum diagonal value.
    For each diagonal value a table is generated ordered by accuracy.
    """
    for metric in metrics:
        min_diag_values = sorted(
            list(
                set(
                    [
                        min_diag_value
                        for method in cluster
                        for min_diag_value in cluster[method]
                    ]
                )
            )
        )

        for min_diag_value in min_diag_values:
            table = PrettyTable()
            table.field_names = ["Method"] + [f"{metric} Mean", f"{metric} Std"]

            # Create a list of rows for this minimum diagonal value
            rows = []
            for method in methods:
                if method in cluster and min_diag_value in cluster[method]:
                    metric_data = cluster[method][min_diag_value]
                    row = [method]
                    if metric in metric_data:
                        row.append(round(metric_data[metric]["mean"], 4))
                        row.append(round(metric_data[metric]["std"], 4))
                    else:
                        row.append("-")
                        row.append("-")
                    rows.append(row)

            # Sort the rows by the mean column in ascending order
            rows.sort(key=lambda row: row[1] if row[1] != "-" else float("inf"))

            # Add the sorted rows to the table
            for row in rows:
                table.add_row([str(value) if value != "-" else value for value in row])

            print(f"\nResults for {metric} (Min Diag Value: {min_diag_value}):")
            print(table)


def average_results(
    exp_name: str = "test_cifar2", path_to_folder: str = "experiments/out"
):
    """
    Given the results on the images experiments, it averages the results in terms of accuracy over different
    experiments with different values of seed.
    Check carefully the path and the experiment name to avoid errors.
    """
    # Initialize dictionaries to store the values
    pretrained_values = {}
    scratch_values = {}

    json_folder = f"{path_to_folder}/{exp_name}/CIFAR-10N"
    if not os.path.isdir(json_folder):
        json_folder = f"{path_to_folder}/{exp_name}/TrashNet"
        if not os.path.isdir(json_folder):
            raise Exception(f"The experiment {exp_name} does not exist!")

    onlyfiles = [
        f
        for f in os.listdir(json_folder)
        if os.path.isfile(os.path.join(json_folder, f))
    ]
    for file in onlyfiles:
        data = read_json(os.path.join(json_folder, file))
        for method, values in data.items():
            if method not in pretrained_values:
                pretrained_values[method] = []
                scratch_values[method] = []

            pretrained_values[method].append(values["pretrained"])
            if values["scratch"] is not None:
                scratch_values[method].append(values["scratch"])
            else:
                scratch_values[method].append(-1)

    # Calculate the average and standard deviation for each method and key
    method_averages = {}
    for method in pretrained_values.keys():
        try:
            pretrained_avg = statistics.mean(pretrained_values[method])
        except TypeError:
            not_pretrained = True
            pretrained_avg = 0
        try:
            scratch_avg = statistics.mean(scratch_values[method])
        except TypeError:
            scratch_avg = 0

        method_averages[method] = (pretrained_avg, scratch_avg)

    # Sort the methods based on the average
    sorted_pretrained = OrderedDict(
        sorted(method_averages.items(), key=lambda x: x[1][0], reverse=True)
    )
    sorted_scratch = OrderedDict(
        sorted(method_averages.items(), key=lambda x: x[1][1], reverse=True)
    )

    # Print the sorted methods

    for method, averages in sorted_pretrained.items():
        if averages[0] != 0.0:
            print(f"Method: {method}, Pretrained Average: {averages[0]:.2f}")
    print()

    for method, averages in sorted_scratch.items():
        if averages[1] != 0.0:
            print(f"Method: {method}, Scratch Average: {averages[1]:.2f}")


def get_directories(path):
    directories = []
    for _, dirs, _ in os.walk(path):
        for directory in dirs:
            directories.append(directory)
    return directories


def obtain_text_results(
    exp_name: str = "new_validation", path_to_folder: str = "experiments/out"
):
    """
    Same as average_results method, but with experiments based on text classification.
    It also returns standard deviation.
    """
    results = {}
    data = []
    parent_folder = f"{path_to_folder}/{exp_name}/"
    languages = get_directories(parent_folder)

    for language in languages:
        json_folder = f"{path_to_folder}/{exp_name}/{language}/"
        onlyfiles = [
            f
            for f in os.listdir(json_folder)
            if os.path.isfile(os.path.join(json_folder, f))
        ]
        results[language] = {}
        for file in onlyfiles:
            output = read_json(os.path.join(json_folder, file))
            for key, subdict in output.items():
                if key not in results[language].keys():
                    results[language][key] = {}
                for metric, value in subdict.items():
                    try:
                        results[language][key][metric].append(value)
                    except KeyError:
                        results[language][key][metric] = []
                        results[language][key][metric].append(value)
    for language, subdict in results.items():
        print(f"************* Language: {language} *************")
        for key, subsubdict in subdict.items():
            print(f"------------- Method: {key} --------------")
            for index, (metric, _) in enumerate(subsubdict.items()):
                std = np.std(np.array(results[language][key][metric]))
                results[language][key][metric] = (
                    np.mean(np.array(results[language][key][metric])),
                    std,
                )
                if index == 0:
                    accuracy_mean = results[language][key][metric][0]
                    accuracy_std = std
                if index == 1:
                    data.append(
                        {
                            "Dataset": language,
                            "Method": key,
                            "Accuracy": str(round(accuracy_mean, 4))
                            + " \u00B1 "
                            + str(round(accuracy_std, 4)),
                            "F1": str(round(results[language][key][metric][0], 4))
                            + " \u00B1 "
                            + str(round(results[language][key][metric][1], 4)),
                        }
                    )
                print(
                    str(metric)
                    + ": "
                    + str(round(results[language][key][metric][0], 4))
                    + " \u00B1 "
                    + str(round(results[language][key][metric][1], 4))
                )

    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv("results.csv", index=False)


def plot_synthetic_exps(experiment_name: str, experiment_path: str, seeds: list = [42]):
    metrics_more_than_one = [
        "dev_loss",
        "test_loss",
        "wrong_predictions_percentage",
    ]
    exp_name = experiment_name
    seeds = seeds
    default_seed = str(seeds[0])
    methods_to_color = {
        "iwmv": "grey",
        "average": "green",
        "posterior": "blue",
        "majority": "violet",
        "dawidskene": "yellow",
        "random": "black",
    }
    prefix = experiment_path
    json_folder = f"{prefix}/{exp_name}/{default_seed}/"
    json_files = sorted(
        [filename for filename in os.listdir(json_folder) if filename.endswith(".json")]
    )
    metrics = None

    cluster = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for filename in json_files:
        for counter, seed in enumerate(seeds, start=1):
            json_folder2 = json_folder.replace(default_seed, str(seed))
            _, method, min_diag_value = (
                filename.replace(default_seed, str(seed))
                .replace(".json", "")
                .split("_")
            )
            res = read_json(json_folder2 + filename.replace(default_seed, str(seed)))[
                min_diag_value
            ][method][0]
            metrics = list(res.keys())
            for metric in metrics:
                cluster[method][min_diag_value][metric]["values"].append(res[metric])
                if metric not in metrics_more_than_one:
                    assert res[metric] <= 1.0
                if len(cluster[method][min_diag_value][metric]["values"]) == 3:
                    cluster[method][min_diag_value][metric]["mean"] = np.mean(
                        cluster[method][min_diag_value][metric]["values"]
                    )
                    cluster[method][min_diag_value][metric]["std"] = np.std(
                        cluster[method][min_diag_value][metric]["values"]
                    )
                    if metric not in metrics_more_than_one:
                        assert cluster[method][min_diag_value][metric]["mean"] <= 1.0

    for metric in metrics:
        minimum_y_value = float("inf")
        ic(metric)
        plt.figure()
        for i, (column_name, color) in enumerate(methods_to_color.items()):
            if column_name in list(cluster.keys()):
                x_axis = list(cluster[column_name].keys())
                y_axis_mean = [
                    value[metric]["mean"] for value in (cluster[column_name].values())
                ]
                y_axis_std = [
                    value[metric]["std"] for value in (cluster[column_name].values())
                ]
                minimum_y_value = min([min(y_axis_mean), minimum_y_value])

                # Calculate width and shift for current column
                width = 0.8 / len(cluster.keys())
                shift = i * width - width / 2

                # Plot histogram
                plt.bar(
                    [x + shift for x in range(len(x_axis))],
                    y_axis_mean,
                    width=width,
                    align="center",
                    label=column_name,
                    linewidth=1.2,
                    color=color,
                )

                plt.errorbar(
                    [x + shift for x in range(len(x_axis))],
                    y_axis_mean,
                    y_axis_std,
                    0.05,
                    linestyle="None",
                    color="black",
                )

        plt.xticks(range(len(x_axis)), x_axis)
        plt.ylim(0.7, 0.99)
        plt.ylabel(metric)
        plt.xlabel("minimum diagonal value T")
        plt.legend()
        plt.savefig(f"{prefix}/{exp_name}/seed_avg_{metric}.png")

    create_results_table(cluster, metrics=["test_acc"], methods=methods_to_color.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description")

    # Define command-line arguments
    parser.add_argument(
        "--plot_type", type=str, default="synthetic", help="Type of the plot"
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--experiment_path",
        default="experiments/out",
        type=str,
        help="Path of the experiment",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44], help="Description of arg1"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.plot_type == "synthetic":
        plot_synthetic_exps(args.experiment_name, args.experiment_path, args.seeds)
    elif args.plot_type == "text":
        obtain_text_results(args.experiment_name)
    elif args.plot_type == "images":
        average_results(exp_name=args.experiment_name)
    else:
        raise Exception(f"Plot type {args.plot_type} not found!")
