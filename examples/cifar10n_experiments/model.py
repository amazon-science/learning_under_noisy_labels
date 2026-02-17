import torch
from torch import nn
import pytorch_lightning as pl
from examples.cifar10n_experiments.utils import get_pretrained_resnet34

# PyTorch Lightning module for CIFAR10 classification
class CIFAR10Model(pl.LightningModule):
    """
    PyTorch Lightning module for CIFAR-10 image classification using a ResNet-34 model.
    """

    def __init__(self, n_classes, pretrained=False):
        """
        Initialize the CIFAR10Model.

        Args:
            n_classes (int): The number of classes in the dataset.
            pretrained (bool, optional): Whether to use a pretrained ResNet-34 model. Defaults to False.
        """
        super().__init__()
        # Initialize the model (ResNet34) and loss function
        self.model = get_pretrained_resnet34(n_classes, pretrained=pretrained)
        self.criterion = nn.CrossEntropyLoss()

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
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        # Define the training step
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning.

        Args:
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        # Define the validation step
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
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
        # Define the test step
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        """
        Configure the optimizer for PyTorch Lightning.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        # Define the optimizer
        return torch.optim.SGD(
            self.parameters(), lr=1e-1, weight_decay=0.0005, momentum=0.9
        )