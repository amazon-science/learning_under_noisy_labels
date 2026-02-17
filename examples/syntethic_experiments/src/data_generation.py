import random
from collections import defaultdict
from typing import *

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.iaa_api import InterAnnotatorAgreementAPI
import matplotlib.pyplot as plt
from examples.cifar10n_experiments.utils import seed_everything


class DataGenerator:
    def __init__(
        self,
        num_classes: int,
        num_samples: int,
        num_features: int,
        num_annotators: int,
        aggregation_type: str,
        leak_distribution: bool,
        equal_distribution: bool,
        prob_non_noise_list: np.array = np.linspace(0.6, 0.9, 4),
        use_auxiliary_aggregation_function: bool = False,
        noise_matrix_type: str = "symmetric",
    ):
        aggregation_methods = [
            "average",
            "posterior",
            "random",
            "majority",
            "mixed",
            "iwmv",
            "dawidskene",
        ]
        assert (
            aggregation_type in aggregation_methods
        ), f"aggregation_type: {aggregation_type} not supported. Use one of them: {str(aggregation_methods)}"
        self.aggregation_type = aggregation_type
        self.num_classes = num_classes
        self.classes = list(range(num_classes))
        self.num_annotators = num_annotators
        self.leak_distribution = leak_distribution
        self.use_auxiliary_aggregation_function = use_auxiliary_aggregation_function

        """
        1. We generate linearly separable data, following a given distribution self.true_distr_y with 0 noise.
        2. We split the dataset into a test split and "train". 
        3. We generate all the possible T given two linear spaces prob_noise_list, prob_non_noise_list.
        4. We create clusters, i.e., we aggregate (computing the mean) all the T-matrix by their minimum value on the diagonal.
        5. generate_noisy_data_for_each_cluster() Given one of the all possible T matrix (e.g., 0.6) you apply noise to the "train" split.
        6. aggregate_data() we aggregate the data following the aggregation_type variable.
        7. We split the noisy train into train and dev (aka validation) set. They will be use to train and tune our NN.
        """

        if num_classes != 4:
            assert (
                equal_distribution
            ), "Currently we don't support data generation with classes higher than four without an equal_distribution"

        self.true_distr_y = (
            np.ones((num_classes)) / num_classes
            if equal_distribution
            else np.array([0.25, 0.35, 0.2, 0.2]) #np.array([0.4, 0.1, 0.4, 0.1]) 
        )

        assert (
            round(self.true_distr_y.sum(), 4) == 1
        ), "the label distribution does not sum up to 1."

        self.data = self.generate_linearly_separable_data(
            num_samples, num_features, self.true_distr_y, **{"random_state": 42}
        )

        self.create_test_split()

        unique_values, distr_y_unique_inverse, distr_y_unique_counts = np.unique(
            self.true_distr_y, return_inverse=True, return_counts=True
        )
        if noise_matrix_type == "symmetric":
            self.all_possible_ts = self.generate_noise_matrices_different_diags(
                prob_non_noise_list, symmetric=True
            )
        if noise_matrix_type == "equal_diagonal_not_symmetric":
            self.all_possible_ts = self.generate_noise_matrices_and_cluster(
                prob_non_noise_list
            )
        if noise_matrix_type == "different_diagonal":
            self.all_possible_ts = self.generate_noise_matrices_different_diags(
                prob_non_noise_list, symmetric=False
            )

        self.noisy_y_given_min_t_value = self.generate_noisy_data_for_each_cluster()

        self.aggregate_data()

        self.split_train_and_dev(percentage=0.2)

    def generate_linearly_separable_data(self, num_samples, num_features, true_distr_y, **kwargs):
        data = {}

        if "random_state" in kwargs:
            np.random.seed(kwargs["random_state"])

        data["x"] = np.random.rand(num_samples, num_features).astype(np.float32)

        app_diff = data["x"][:, 0] - data["x"][:, 1]
        app_argsort = np.argsort(app_diff)

        class_until_idx = np.array(np.cumsum(true_distr_y) * num_samples, dtype=int)
        class_until_idx[-1] = num_samples

        data["true_y"] = np.zeros(num_samples).astype(np.float32)
        i0 = class_until_idx[0]
        for c, i1 in enumerate(class_until_idx[1:]):
            data["true_y"][app_argsort[i0:i1]] = c + 1
            i0 = i1

        return data

    def plot_obtained_data(self, num_samples, num_features, true_distr_y):
        data = self.generate_linearly_separable_data(
            num_samples, num_features, true_distr_y
        )
        X, y = data["x"], data["true_y"]
        colors = ["r", "g", "b", "c"]

        plt.figure(figsize=(8, 6))
        for label in range(4):
            plt.scatter(
                X[y == label, 0],
                X[y == label, 1],
                c=colors[label],
                label=f"Class {label}",
                alpha=0.5,
            )

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Linearly Separable Data")
        plt.legend()
        plt.savefig("a.png", format="png")
        plt.show()

    def create_test_split(self, test_size=0.20) -> None:
        """
        This function splits the ground truth data into gold train and gold test.
        Then we will split the training portion into a dev.
        """
        data_train, data_test, labels_train, labels_test = train_test_split(
            self.data["x"], self.data["true_y"], test_size=test_size, random_state=42
        )
        self.train_features = data_train
        self.gold_labels_train = labels_train
        self.test_features = data_test
        self.gold_test_labels = labels_test
        self.data["x"] = data_train
        self.data["true_y"] = labels_train

    def generate_numbers(self, diagonal_entry):
        """This method generates a T matrix with all the same diagonal entry.
        The values outside the diagonal can change and they are created
        such that each rows sums up to 1.
        """
        results = diagonal_entry * np.eye(self.num_classes)
        for index_row in range(self.num_classes - 1):
            remaining_sum = 1 - diagonal_entry
            for index_col, _ in enumerate(results[index_row]):
                if index_row != index_col:
                    if index_col < self.num_classes - 1:
                        random_number = random.uniform(0, remaining_sum)
                        results[index_row][index_col] = random_number
                        remaining_sum -= random_number
                    else:
                        results[index_row][index_col] = remaining_sum

        results[self.num_classes - 1][self.num_classes - 1] = diagonal_entry
        last_row_sum = 1 - diagonal_entry
        for index_col in range(self.num_classes - 1):
            if index_col < self.num_classes - 2:
                random_number = random.uniform(0, last_row_sum)
                results[self.num_classes - 1][index_col] = random_number
                last_row_sum -= random_number
            else:
                results[self.num_classes - 1][index_col] = last_row_sum
        return results

    def generate_noise_matrices_and_cluster(
        self, prob_non_noise_list, num_matrices: int = 50, diag_variation: float = 0.05
    ):
        """
        For each diagonal value in prob_non_noise_list, num_matrices matrices are created with
        diagonal value between:
        prob_non_noise_list[i] - diag_variation and prob_non_noise_list[i] + diag_variation
        Then for each diagonal value the matrices are collected and then averaged.
        """
        self.all_possible_ts = {key: [] for key in prob_non_noise_list}

        for prob in prob_non_noise_list:
            interval = (
                (prob + diag_variation) - (prob - diag_variation)
            ) / num_matrices
            for i in np.arange(prob - diag_variation, prob + diag_variation, interval):
                T_matrix = self.generate_numbers(i)
                self.all_possible_ts[prob].append(T_matrix)

            matrices_array = np.array(self.all_possible_ts[prob])
            self.all_possible_ts[prob] = np.mean(matrices_array, axis=0)
        return self.all_possible_ts

    def generate_symmetric_matrix(self, diagonal_entry):
        """
        Generate symmetric T matrices with the same diagonal entries so that rows and cols sum to 1
        """
        values = [0.6, 0.7, 0.8, 0.9]
        matrix = np.diag(values)
        # matrix = diagonal_entry * np.eye(self.num_classes)

        for i in range(self.num_classes):
            remaining = 1 - matrix[i][i]
            off_diag_indices = [j for j in range(self.num_classes) if j != i]
            values = np.random.dirichlet(np.ones(self.num_classes - 1)) * remaining
            matrix[i, off_diag_indices] = values

        # Adjust columns to sum to 1
        for j in range(self.num_classes):
            col_sum = np.sum(matrix[:, j])
            if col_sum != 1:
                for i in range(self.num_classes):
                    if i == j:
                        continue
                    matrix[i][j] += (1 - col_sum) / (self.num_classes - 1)

        # Avoid negative values
        matrix = np.maximum(matrix, 0)

        # Normalize rows and columns to ensure double stochasticity
        for i in range(self.num_classes):
            matrix[i] /= np.sum(matrix[i])
        for j in range(self.num_classes):
            matrix[:, j] /= np.sum(matrix[:, j])

        matrix = (matrix + matrix.T) / 2
        return matrix

    def generate_symmetric_matrices_and_cluster(
        self, prob_non_noise_list, num_matrices: int = 50
    ):
        """
        For each diagonal value in prob_non_noise_list, num_matrices matrices are created with
        diagonal value given by the item in prob_non_noise_list
        Then for each diagonal value the matrices are collected and then averaged.
        """
        self.all_possible_ts = {key: [] for key in prob_non_noise_list}

        for prob in prob_non_noise_list:
            for _ in range(num_matrices):
                T_matrix = self.generate_symmetric_matrix(prob)
                self.all_possible_ts[prob].append(T_matrix)

            matrices_array = np.array(self.all_possible_ts[prob])
            self.all_possible_ts[prob] = np.mean(matrices_array, axis=0)

        return self.all_possible_ts

    def matrices_different_diag(self, prob_non_noise_list):
        """This method generates a T matrix with all the same diagonal entry.
        The values outside the diagonal can change and they are created
        such that each rows sums up to 1.
        """
        results = np.eye(self.num_classes)
        diagonal_values = []
        for index_row in range(self.num_classes - 1):
            diagonal_entry = random.choice(prob_non_noise_list)
            diagonal_values.append(diagonal_entry)
            results[index_row][index_row] = diagonal_entry
            remaining_sum = 1 - diagonal_entry
            for index_col, _ in enumerate(results[index_row]):
                if index_row != index_col:
                    if index_col < self.num_classes - 1:
                        random_number = random.uniform(0, remaining_sum)
                        results[index_row][index_col] = random_number
                        remaining_sum -= random_number
                    else:
                        results[index_row][index_col] = remaining_sum

        results[self.num_classes - 1][self.num_classes - 1] = diagonal_entry
        last_row_sum = 1 - diagonal_entry
        for index_col in range(self.num_classes - 1):
            if index_col < self.num_classes - 2:
                random_number = random.uniform(0, last_row_sum)
                results[self.num_classes - 1][index_col] = random_number
                last_row_sum -= random_number
            else:
                results[self.num_classes - 1][index_col] = last_row_sum
        return results, min(diagonal_values)

    def define_T_symmetric_and_double_stoc(
        self, prob_non_noise_list, diag_entries=None
    ):
        diag_values = []
        if diag_entries is None:
            for _ in range(self.num_classes):
                diagonal_entry = random.choice(prob_non_noise_list)
                diag_values.append(diagonal_entry)
        else:
            for _ in range(self.num_classes):
                diag_values.append(diag_entries)
        sorted_diag_indices = np.argsort(diag_values)[::-1]
        sorted_diag_values = [diag_values[j] for j in sorted_diag_indices]

        # we first work with a matrix with values sorted on the diagonal
        T = np.diag(sorted_diag_values)
        for i in range(self.num_classes - 3):
            sum_non_zero_elements = max(1 - np.sum(T[i]), 0)
            values = np.random.rand(self.num_classes - i - 2)
            random_value = sum_non_zero_elements * np.random.rand(1)
            values = random_value * values / np.sum(values)
            indices = np.arange(i + 1, self.num_classes - 1)
            T[i, indices] = values
            T[indices, i] = values
        sum = np.sum(
            np.sum(T[0 : self.num_classes - 1, 0 : self.num_classes - 1], axis=0)
        )
        x = (self.num_classes - 1 - sum - 1 + sorted_diag_values[-1]) / 2
        T[self.num_classes - 3, self.num_classes - 2] = x
        T[self.num_classes - 2, self.num_classes - 3] = x
        T[self.num_classes - 1, 0 : self.num_classes - 1] = np.ones(
            self.num_classes - 1
        ) - np.sum(T[0 : self.num_classes - 1, 0 : self.num_classes - 1], axis=0)
        T[0 : self.num_classes - 1, self.num_classes - 1] = np.ones(
            self.num_classes - 1
        ) - np.sum(T[0 : self.num_classes - 1, 0 : self.num_classes - 1], axis=1)

        # Now we apply the permutation matrix to permute rows and columns and come back to the original order of the diag
        P = np.zeros_like(T)
        for i, j in enumerate(sorted_diag_indices):
            P[i, j] = 1

        T_permuted = P.T @ T @ P

        return T_permuted, min(diag_values)

    def generate_noise_matrices_different_diags(
        self, prob_non_noise_list, num_matrices: int = 10, symmetric=False
    ):
        """
        Generate num_matrices T matrices wuth different diagonal values and then cluster them
        on the basis of the minimum diagonal value.
        Remember to give a value to num_matrices high enough.
        Otherwise, it will be difficult to generate matrices with 0.9 as min value.
        """
        self.all_possible_ts = {key: [] for key in prob_non_noise_list}
        if not symmetric:
            prob_non_noise_list = np.append(prob_non_noise_list, 1.0)

        for noise_value in self.all_possible_ts.values():
            while len(noise_value) < num_matrices:
                if not symmetric:
                    T_matrix, min_value = self.matrices_different_diag(
                        prob_non_noise_list
                    )
                    self.all_possible_ts[min_value].append(T_matrix)
                else:

                    T_matrix, min_value = self.define_T_symmetric_and_double_stoc(
                        prob_non_noise_list
                    )

                    if min_value != 1.0:
                        self.all_possible_ts[min_value].append(T_matrix)

        for prob in self.all_possible_ts.keys():
            matrices_array = np.array(self.all_possible_ts[prob])
            self.all_possible_ts[prob] = np.mean(matrices_array, axis=0)
        return self.all_possible_ts

    def noisify_y(self, T):
        N = self.data["true_y"].shape[0]
        noisy_Y = np.zeros((N, self.num_annotators), dtype=int)  # annotator labels
        for i in range(N):
            try:
                noisy_Y[i] = np.random.choice(
                    self.num_classes,
                    self.num_annotators,
                    p=T[int(self.data["true_y"][i])],
                )
            except ValueError:
                sum = np.sum(T[int(self.data["true_y"][i])])
                remaining = 1 - sum
                T[int(self.data["true_y"][i])][0] += remaining

        return noisy_Y

    def generate_noisy_data_for_each_cluster(self):
        self.noisy_y_given_min_t_value = {
            min_t_diag_value: self.noisify_y(T=gold_t_matrix).astype(np.int32)
            for min_t_diag_value, gold_t_matrix in self.all_possible_ts.items()
        }
        return self.noisy_y_given_min_t_value

    def aggregate_data(self) -> None:
        """
        This function aggregates the noisy data, accordingly to the aggregation method (posterior, random, etc) in the constructor.
        It also collect all the estimated T matricies (t_hat_matrix), they are useful to compute the MSE between the gold T and T hat.
        """
        self.aggregated_noisy_data, self.collect_estimated_t_matrices = {}, {}

        for min_t_diag_value, noisy_y in self.noisy_y_given_min_t_value.items():
            aggregated_annotations, t_hat_matrix = self.aggregate_annotations(noisy_y)
            self.collect_estimated_t_matrices[min_t_diag_value] = t_hat_matrix
            self.aggregated_noisy_data[min_t_diag_value] = aggregated_annotations

    def split_train_and_dev(self, percentage: float) -> None:
        """
        This function splits the data into noisy train and noisy dev (or validation set).
        In that way we can train and optimizing our network on aggregated data rather than on gold data that are not avaiable in our scenario.
        """
        self.data_given_min_t_value = {}
        self.aggregated_train_data_given_min_t_value = {}
        self.aggregated_dev_data_given_min_t_value = {}
        for min_t_diag_value, noisy_y in self.aggregated_noisy_data.items():
            print(f"Min diag value: {min_t_diag_value}")
            assert self.train_features.shape[0] == noisy_y.shape[0], (
                self.train_features.shape[0],
                noisy_y.shape[0],
            )
            data_train, data_dev, labels_train, labels_dev = train_test_split(
                self.train_features, noisy_y, test_size=percentage, random_state=42
            )
            self.aggregated_train_data_given_min_t_value[min_t_diag_value] = {
                "data": data_train,
                "labels": labels_train,
            }
            self.aggregated_dev_data_given_min_t_value[min_t_diag_value] = {
                "data": data_dev,
                "labels": labels_dev,
            }

    def aggregate_annotations(
        self, annotations: np.array
    ) -> Tuple[torch.Tensor, Union[None, np.array]]:
        api = InterAnnotatorAgreementAPI(
            annotations,
            label_distribution=(
                self.true_distr_y.tolist() if self.leak_distribution else None
            ),
        )
        if self.aggregation_type == "posterior":
            return api.get_posterior_probability(), api.t_hat
        if self.aggregation_type == "dawidskene":
            return api.get_dawid_skene_labels_from_annotations(), api._t_hat
        if self.aggregation_type == "iwmv":
            return api.get_iwmv_labels_from_annotations(), api._t_hat
        elif self.aggregation_type == "majority":
            return api.get_hard_voting_annotations(), None
        elif self.aggregation_type == "random":
            return api.get_random_labels_from_anxxnotations(), None
        elif self.aggregation_type == "average":
            return api.get_average_soft_labels(), None
        elif self.aggregation_type == "mixed":
            return api.get_mixed(), api.t_hat

    def collate_noisy_annotations(self, stage: str, min_t_diag_value: float):
        """
        This function is used in get_data in order to retrieve and format training and validation data.
        The formatting allows for a dataloader
        """
        assert stage in ["train", "dev"]
        if stage == "train":
            dataset = self.aggregated_train_data_given_min_t_value[min_t_diag_value]
        else:
            dataset = self.aggregated_dev_data_given_min_t_value[min_t_diag_value]
        return [
            [feature, label]
            for feature, label in zip(dataset["data"], dataset["labels"])
        ]

    def get_data(self, stage: str, min_t_diag_value: float = None):
        """
        This function it is used to provide data already aggregated for each . Given a min_t_diag_value.
        """
        assert stage in ["train", "dev", "test"]
        assert (
            min_t_diag_value is not None
            and stage in ["train", "dev"]
            and 0.0 <= min_t_diag_value <= 1
        ) or (min_t_diag_value is None and stage != "train")
        if stage in ["train", "dev"]:
            return self.collate_noisy_annotations(stage, min_t_diag_value)
        else:
            assert self.test_features.shape[0] == self.gold_test_labels.shape[0]
            return [
                [x, int(y)] for x, y in zip(self.test_features, self.gold_test_labels)
            ]


if __name__ == "__main__":

    seed_everything(42)

    dg = DataGenerator(
        num_classes=4,
        num_features=2,
        num_samples=10000,
        num_annotators=3,
        aggregation_type="posterior",
        leak_distribution=False,
        equal_distribution=False,
    )
    dg.plot_obtained_data(10000, 2, np.array([0.4, 0.1, 0.4, 0.1]))
