import numpy as np
import matplotlib.pyplot as plt
from examples.syntethic_experiments.src.data_generation import DataGenerator
from tqdm import tqdm
import pickle
import matplotlib.lines as mlines
from examples.cifar10n_experiments.utils import seed_everything


def accuracy(list1, list2, aggregation_type):
    """
    Simple method to compute the accuracy.
    """
    if aggregation_type in ["posterior", "average"]:
        list2 = np.argmax(list2, axis=1).tolist()
    total = 0
    for a, b in zip(list1, list2):
        if a == b:
            total += 1
    return total / len(list1)


def define_legend(data):
    """
    Define the legend, markers and colors for the plot.
    """
    agg_to_real_name = {
        "posterior": "Posterior",
        "dawidskene": "Dawid-Skene",
        "iwmv": "IWMV",
    }

    res = []
    methods = list(data.keys())
    probs = list(list(data.values())[0].values())[0].keys()
    linestyles = obtain_linestyles(len(probs)) 
    colors = obtain_color(len(methods)) 
    for i, method in enumerate(methods):
        res.append(
            mlines.Line2D(
                [],
                [],
                color=colors[i],
                marker=".",
                linestyle="None",
                markersize=10,
                label=agg_to_real_name[method],
            )
        )

    for j, prob in enumerate(probs):
        res.append(
            mlines.Line2D(
                [], [], color="black", ls=linestyles[j], markersize=10, label=f"{prob}"
            )
        )
    return res


def obtain_linestyles(number_of_styles):
    list_of_styles = ["solid", "dashed", "dashdot", "dotted"]
    return list_of_styles[:number_of_styles]


def obtain_markers(number_of_markers):
    list_of_markers = ["X", "v", "p", "*", "P", "s", "+", "o", "D", "x"]
    return list_of_markers[:number_of_markers]


def obtain_color(number_of_colors):
    all = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#984ea3",
        "#377eb8",
        "#a65628",
        "#999999",
        "#dede00",
    ]
    return all[:number_of_colors]


def generate_plot(
    num_samples: np.array,
    desired_noise_value: float = 0.7,
    num_classes: int = 4,
    equal_distribution: bool = True,
    num_annotators: int = 3,
    num_features: int = 2,
    mutiple_eigs: bool = False,
    discard_values: np.array = None,
    seed: int = 42,
) -> dict:
    """
    This method generates a plot where on the x-axis it is possible to find the number of samples.
    There are two y-axes. On the first one it can be found the estimation error.
    One the second one the accuracy.
    """
    
    seed_everything(seed)

    if discard_values is not None and not mutiple_eigs:
        raise ValueError(
            "You need to have multiple minimum diagonal values to discard some of them!"
        )

    results = {}
    aggregation_results = {}
    aggregations = np.array(["dawidskene", "posterior", "iwmv"])
    for aggregation_type in tqdm(aggregations):
        results[aggregation_type] = {}
        aggregation_results[aggregation_type] = {}
        for num_sample in tqdm(num_samples):
            results[aggregation_type][num_sample] = {}
            aggregation_results[aggregation_type][num_sample] = {}
            data_generator = DataGenerator(
                num_classes=num_classes,
                num_samples=num_sample,
                num_features=num_features,
                num_annotators=num_annotators,
                aggregation_type=aggregation_type,
                leak_distribution=False,
                equal_distribution=equal_distribution,
            )
            for key, _ in data_generator.collect_estimated_t_matrices.items():
                if not mutiple_eigs:
                    if key == desired_noise_value:
                        results[aggregation_type][num_sample] = np.linalg.norm(
                            data_generator.all_possible_ts[key]
                            - data_generator.collect_estimated_t_matrices[key],
                            ord="fro",
                        )
                        aggregation_results[aggregation_type][num_sample] = accuracy(
                            data_generator.data["true_y"],
                            data_generator.aggregated_noisy_data[key],
                            aggregation_type,
                        )
                else:
                    if discard_values is not None:
                        if key not in discard_values:
                            results[aggregation_type][num_sample][key] = np.linalg.norm(
                                data_generator.all_possible_ts[key]
                                - data_generator.collect_estimated_t_matrices[key],
                                ord="fro",
                            )
                            aggregation_results[aggregation_type][num_sample][key] = (
                                accuracy(
                                    data_generator.data["true_y"],
                                    data_generator.aggregated_noisy_data[key],
                                    aggregation_type,
                                )
                            )
    with open(f"results_norm_plot_{seed}.pkl", "wb") as file:
        pickle.dump(results, file)
    with open(f"aggregation_results_norm_plot_{seed}.pkl", "wb") as file:
        pickle.dump(aggregation_results, file)
    return results, aggregation_results, aggregations

def plot_results(
    data_list: list = None,
    aggregation_results_list: list = None,
    aggregations: np.array = None,
    class_distributions: list = [0.6, 0.8],
    load: bool = False,
    multiple_eigs: bool = False,
    seed_values: list = [42, 43, 44, 45, 46]
):
    """
    Plot comparison results for different aggregation methods.
    
    Args:
        data_list (list): List of dictionaries containing results for each seed
        aggregation_results_list (list): List of aggregation results
        aggregations (np.array): Array of aggregation method names
        class_distributions (list): List of class distribution values to plot
        load (bool): Whether to load data from files
        multiple_eigs (bool): Whether to plot multiple eigenvalues
        seed_values (list): List of random seeds used
    """
    # Method name mapping
    agg_to_real_name = {
        "posterior": "Posterior",
        "dawidskene": "Dawid-Skene",
        "iwmv": "IWMV",
    }
    
    # Load data if required
    if load or (data_list is None and aggregations is None):
        data_list = []
        for seed in seed_values:
            with open(f"results_norm_plot_{seed}.pkl", "rb") as file:
                data_list.append(pickle.load(file))
        aggregations = np.array(list(agg_to_real_name.keys()))

    # Create subplots
    fig, axes = plt.subplots(1, len(class_distributions), figsize=(15, 5))
    
    if not multiple_eigs:
        colors = obtain_color(len(aggregations))
        x_values = list(list(data_list[0].values())[0].keys())
        
        # Plot for each class distribution
        for idx, class_dist in enumerate(class_distributions):
            ax = axes[idx]
            ax2 = ax.twinx() if aggregation_results_list is not None else None
            
            # Plot for each aggregation method
            for i, aggregation in enumerate(aggregations):
                # Calculate mean and standard deviation
                y_values = []
                y_stds = []
                
                for x in x_values:
                    values = [data[aggregation][x][class_dist] for data in data_list]
                    y_values.append(np.mean(values))
                    y_stds.append(np.std(values))
                
                # Plot main line and confidence band
                ax.plot(x_values, y_values, label=agg_to_real_name[aggregation], color=colors[i])
                ax.fill_between(x_values, 
                              np.array(y_values) - np.array(y_stds),
                              np.array(y_values) + np.array(y_stds),
                              alpha=0.2, color=colors[i])
                
                # Plot aggregation results if available
                if aggregation_results_list is not None:
                    y_values_agg = []
                    y_stds_agg = []
                    for x in x_values:
                        values_agg = [agg_res[aggregation][x][class_dist] 
                                    for agg_res in aggregation_results_list]
                        y_values_agg.append(np.mean(values_agg))
                        y_stds_agg.append(np.std(values_agg))
                    
                    ax.plot(x_values, y_values_agg, color=colors[i])
                    ax.fill_between(x_values,
                                  np.array(y_values_agg) - np.array(y_stds_agg),
                                  np.array(y_values_agg) + np.array(y_stds_agg),
                                  alpha=0.2, color=colors[i])

            # Set axis labels and properties
            ax.set_xlabel("Number of samples")
            if aggregation_results_list is not None:
                ax2.set_ylabel("Accuracy")
            ax.set_ylabel(r"$\|T - \hat{T}\|_2$")
            ax.set_xscale("log")
            
            # Configure x-axis ticks
            ax.xaxis.set_ticks([], minor=True)
            labels = [r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"]
            ax.set_xticks([10**i for i in range(1, len(labels) + 1)], labels)
            
            # Set title based on class distribution
            if class_dist == 0.8:
                ax.set_title(r"$\mathcal{D}=(0.8,0.2)$")
            if class_dist == 0.6:
                ax.set_title(r"$\mathcal{D}=(0.6,0.4)$")
            
            # Add legend to first plot only
            if idx == 0:
                plt.subplots_adjust(
                    bottom=0.25, wspace=0.25,
                    left=0.05, right=0.95
                )
                ax.legend(bbox_to_anchor=(1, -0.39), ncol=3, loc='lower center')

    # Save the plot
    plt.savefig("results_comparison.png", format="png")

if __name__ == "__main__":
    # Set matplotlib parameters
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15
    })
    
    # Load aggregation results
    aggregation_results = []
    for seed in [42, 43, 44, 45, 46]:
        with open(f"aggregation_results_norm_plot_{seed}.pkl", "rb") as file:
            aggregation_results.append(pickle.load(file))
            
    # Generate plots
    plot_results(
        aggregations=np.array(["dawidskene", "posterior", "iwmv"]),
        load=True,
        aggregation_results_list=aggregation_results
    )