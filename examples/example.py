import warnings

import numpy as np
import torch
from src.iaa_api import InterAnnotatorAgreementAPI

warnings.filterwarnings("ignore", category=UserWarning)


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_everything(42)
    loss_func = torch.nn.CrossEntropyLoss()

    human_annotations = np.array([[0, 1, 2, 3, 4], [0, 2, 2, 0, 4], [0, 2, 1, 0, 4]]).T

    # 4 examples, 3 annotators and 3 classes
    api = InterAnnotatorAgreementAPI(human_annotations)

    model_predictions = torch.randn(api.num_samples, api.num_classes, requires_grad=True)
    # dataset_size, num_classes

    # Posterior probability example
    posterior_probability = api.get_posterior_probability(return_torch=True)
    loss = loss_func(model_predictions, posterior_probability)
    loss.backward()

    # Average soft label example
    average_soft_labels = api.get_average_soft_labels(return_torch=True)
    loss = loss_func(model_predictions, average_soft_labels)
    print(loss)
    loss.backward()

    # Monodimensional labels
    hard_voting_annotations = api.get_hard_voting_annotations(return_torch=True)
    loss = loss_func(model_predictions, hard_voting_annotations)
    print(loss)
    loss.backward()

    random_labels = api.get_random_labels(return_torch=True)
    loss = loss_func(model_predictions, random_labels)
    print(loss)
    loss.backward()

    random_labels_from_annotations = api.get_random_labels_from_annotations(return_torch=True)
    loss = loss_func(model_predictions, random_labels_from_annotations)
    print(loss)
    loss.backward()
