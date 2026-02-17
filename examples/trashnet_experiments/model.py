from typing import *
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from examples.cifar10n_experiments.utils import get_pretrained_resnet34
from src.iaa_api import InterAnnotatorAgreementAPI

# Define the PyTorch Lightning module
class TrashNetClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for TrashNet image classification using a ResNet-34 model.
    """

    def __init__(self, n_classes, pretrained, aggregation_type, lr=2e-3):
        """
        Initialize the TrashNetClassifier module.

        Args:
            n_classes (int): The number of classes in the dataset.
            pretrained (bool): Whether to use a pretrained ResNet-34 model.
            aggregation_type (str): The type of annotation aggregation to use.
            lr (float, optional): The learning rate for the optimizer. Defaults to 2e-3.
        """
        super().__init__()
        self.model = get_pretrained_resnet34(n_classes, pretrained=pretrained)
        self.criterion = nn.CrossEntropyLoss()
        self.aggregation_type = aggregation_type
        self.lr = lr

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output logits.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.

        Args:
            batch (tuple): A tuple containing the input data and annotations.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        inputs, _, annotations = batch
        annotations = np.array(annotations.tolist())
        api = InterAnnotatorAgreementAPI(annotations)
        labels = api.return_annotations(self.aggregation_type).to(self.device)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning.

        Args:
            batch (tuple): A tuple containing the input data and annotations.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        inputs, _, annotations = batch
        annotations = np.array(annotations.tolist())
        api = InterAnnotatorAgreementAPI(annotations)
        labels = api.return_annotations(self.aggregation_type).to(self.device)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Testing step for PyTorch Lightning.

        Args:
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the loss and accuracy for the current batch.
        """
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        accuracy = (outputs.argmax(1) == labels).float().mean()
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', accuracy, on_epoch=True)
        return {'test_loss': loss, 'test_acc': accuracy}

    def configure_optimizers(self):
        """
        Configure the optimizer for PyTorch Lightning.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)