import json
import torchvision
import torch
from torch import nn
from multiprocessing import cpu_count
import numpy as np 
from src.iaa_api import InterAnnotatorAgreementAPI
from torch.utils.data import random_split


def seed_everything(seed: int):
    """
    Set the random seed for Python's random module, NumPy, PyTorch, and PyTorch Lightning.

    Args:
        seed (int): The random seed value.

    This function ensures reproducibility across different runs by setting the same random seed
    for various libraries and frameworks used in the code.
    """
    import random, os
    import numpy as np
    import torch
    import pytorch_lightning as pl

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    pl.seed_everything(seed)


def get_number_of_cpu_cores() -> int:
    """
    Return the number of CPU cores available on the system.

    Returns:
        int: The number of CPU cores.
    """
    return cpu_count()


def get_torch_device() -> torch.device:
    """
    Return the appropriate PyTorch device based on the availability of CUDA.

    Returns:
        torch.device: The PyTorch device ('cuda' if available, 'cpu' otherwise).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def read_json(filename: str) -> dict:
    """
    Read a JSON file and return its contents as a Python dictionary.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a Python dictionary.
    """
    with open(filename, "r") as reader:
        return json.load(reader)


def write_json(filename: str, dictionary: dict) -> None:
    """
    Write a Python dictionary to a JSON file with indentation for better readability.

    Args:
        filename (str): The path to the JSON file.
        dictionary (dict): The Python dictionary to be written to the JSON file.
    """
    with open(filename, "w") as writer:
        json.dump(dictionary, writer, indent=4)


def _freeze_parameters(model: nn.Module) -> nn.Module:
    """
    Freeze the parameters of a PyTorch model by setting the `requires_grad` attribute to False for all parameters.

    Args:
        model (nn.Module): The PyTorch model whose parameters need to be frozen.

    Returns:
        nn.Module: The model with frozen parameters.

    Note: This function is used internally and is not intended for direct use.
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_pretrained_resnet34(n_classes: int, pretrained: bool) -> nn.Module:
    """
    Return a PyTorch ResNet34 model with the specified number of classes.

    Args:
        n_classes (int): The number of classes for the classification task.
        pretrained (bool): Whether to initialize the model with pretrained weights from torchvision.

    Returns:
        nn.Module: The ResNet34 model with the final fully connected layer replaced with a new linear layer
                   with the specified number of classes.

    If `pretrained` is True, the model is initialized with pretrained weights from torchvision.
    The final fully connected layer is replaced with a new linear layer with the specified number of classes.
    """
    weights = None if not pretrained else torchvision.models.ResNet34_Weights.DEFAULT
    model = torchvision.models.resnet34(weights=weights)
    # model = _freeze_parameters(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, n_classes),
    )

    return model
def get_percentage(total, percentage):
    """
    Calculate a percentage of a total value.

    Args:
        total (int or float): The total value.
        percentage (float): The percentage value between 0 and 1.

    Returns:
        float: The calculated percentage of the total value.

    Raises:
        AssertionError: If the percentage is not between 0 and 1.
    """
    assert 0 <= percentage <= 1.0, "Percentage should be in the scale of 0 to 1"
    return total - (total * (1 - percentage))


# Function to split the dataset into train and validation sets
def get_train_val_split(dataset, seed=21094):
    """
    Split a PyTorch dataset into train and validation sets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        seed (int, optional): The random seed for reproducibility. Defaults to 21094.

    Returns:
        tuple: A tuple containing the train set, train data, validation set, validation data,
               train indices, and validation indices.
    """
    train_size, val_size = int(get_percentage(len(dataset), 0.80)), int(
        get_percentage(len(dataset), 0.20)
    )
    train_set, val_set = random_split(
        dataset, [train_size, val_size], torch.Generator().manual_seed(seed)
    )

    # Extract data and targets for train and validation sets
    train_targets = [train_set.dataset.targets[i] for i in train_set.indices]
    train_data = [train_set.dataset.data[i] for i in train_set.indices]
    val_targets = [val_set.dataset.targets[i] for i in val_set.indices]
    val_data = [val_set.dataset.data[i] for i in val_set.indices]

    train_set.targets = train_targets
    val_set.targets = val_targets
    train_set.data = train_data
    val_set.data = val_data
    return train_set, train_data, val_set, val_data, train_set.indices, val_set.indices


# Function to load CIFAR10N dataset with noisy annotations
def load_cifar10n(aggregation_type, dataset):
    """
    Load the CIFAR-10N dataset with noisy annotations.

    Args:
        aggregation_type (str): The type of annotation aggregation to use.
        dataset (torch.utils.data.Dataset): The CIFAR-10 dataset.

    Returns:
        list: A list of aggregated labels for the dataset.
    """
    noise_file = torch.load("examples/cifar10n_experiments/data/CIFAR-10_human.pt")
    assert torch.equal(
        torch.Tensor(dataset.targets), torch.Tensor(noise_file["clean_label"])
    )

    random_label1 = noise_file["random_label1"]
    random_label2 = noise_file["random_label2"]
    random_label3 = noise_file["random_label3"]

    annotations = np.vstack([random_label1, random_label2, random_label3]).T
    api = InterAnnotatorAgreementAPI(annotations)
    labels = api.return_annotations(aggregation_type)
    return labels