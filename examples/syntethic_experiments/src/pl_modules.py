from typing import Any, Dict, List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics

from examples.syntethic_experiments.src.nn_model import (FeedForwardNN,
                                                         LogisticRegression
                                                         )

class BasePLModule(pl.LightningModule):
    """
    Base PyTorch Lightning module for classification tasks.
    """
    def __init__(self, conf: Optional[omegaconf.DictConfig] = None) -> None:
        """
        Initialize the BasePLModule.

        Args:
            conf (Optional[omegaconf.DictConfig], optional): Configuration object. Defaults to None.
        """
        super().__init__()
        self.conf: omegaconf.DictConfig = conf
        self.save_hyperparameters(conf)
        if conf.data.num_classes == 2:
            self.model = LogisticRegression(conf.data.num_features, conf.model.hidden_dim, conf.data.num_classes)
        else:
            self.model = FeedForwardNN(conf.data.num_features, conf.model.hidden_dim, conf.data.num_classes)
        self.loss_func = torch.nn.functional.cross_entropy
        self.accuracy: Dict[torchmetrics.Accuracy] = {
            "train": torchmetrics.Accuracy(task="multiclass", num_classes=conf.data.num_classes),
            "val": torchmetrics.Accuracy(task="multiclass", num_classes=conf.data.num_classes),
            "test": torchmetrics.Accuracy(task="multiclass", num_classes=conf.data.num_classes),
        }
        self.confidence_threshold = 0.35
        self.total_errors = {'val' : 0, 'test' : 0}

    def forward(self, input_vector) -> dict:
        """
        Forward pass of the model.

        Args:
            input_vector (torch.Tensor): Input features.

        Returns:
            dict: Logits from the model.
        """
        logits = self.model(input_vector)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step for PyTorch Lightning.

        Args:
            batch (dict): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, y = batch
        if self.conf.data.aggregation_type in ["random", "majority"]:
            y = y.type(torch.LongTensor).to(self.device)
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        return {"loss": loss}

    def _shared_step(self, batch, phase: str):
        """
        Shared step for validation and testing.

        Args:
            batch (dict): Input batch containing features and labels.
            phase (str): Phase of the step, either "val" or "test".

        Returns:
            dict: Dictionary containing loss, accuracy, and F1 score.
        """
        assert phase in ["val", "test"]
        x, y = batch

        logits = self.forward(x)

        if y.dtype == torch.int32:
            y = y.to(torch.long)

        loss = self.loss_func(logits, y)

        if phase == "val":
            return {f"{phase}_loss": loss}

        y_preds = torch.argmax(logits, dim=-1)
        acc = torchmetrics.functional.accuracy(
            y_preds.cpu(), y.cpu(), task="multiclass", num_classes=self.conf.data.num_classes
        )
        f1 = torchmetrics.functional.f1_score(
            y_preds.cpu(), y.cpu(), task="multiclass", num_classes=self.conf.data.num_classes, average="macro"
        )

        self.log(f"{phase}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_f1", f1, on_epoch=True, prog_bar=True)

        if self.confidence_threshold is not None:
            wrong_predictions = self.compute_wrong_predictions(logits, y)
            self.total_errors[phase] += wrong_predictions

        return {f"{phase}_loss": loss, f"{phase}_acc": acc}

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """
        Validation step for PyTorch Lightning.

        Args:
            batch (dict): Input batch containing features and labels.
            batch_idx (int): Index of the batch.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        """
        Testing step for PyTorch Lightning.

        Args:
            batch (dict): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Any: Dictionary containing loss and accuracy.
        """
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """
        Configure optimizers for PyTorch Lightning.

        Returns:
            Any: Optimizer instance.
        """
        return hydra.utils.instantiate(self.conf.model.optimizer, params=self.parameters())

    def compute_wrong_predictions(self, predictions, labels):
        """
        Compute the number of wrong predictions with high confidence.

        Args:
            predictions (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            int: Number of wrong predictions with high confidence.
        """
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        confidences, predicted_classes = torch.max(probabilities, 1)
        incorrect_high_confidence = (predicted_classes != labels) & (confidences > self.confidence_threshold)
        num_wrong_predictions = torch.sum(incorrect_high_confidence).item()
        return num_wrong_predictions