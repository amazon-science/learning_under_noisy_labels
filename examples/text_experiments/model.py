from transformers import XLMRobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import torch
import pytorch_lightning as pl

class TextClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for text classification using XLM-RoBERTa.
    """

    def __init__(self, num_labels, language):
        """
        Initialize the TextClassifier module.

        Args:
            num_labels (int): The number of output labels for the classification task.
            language (str): The language of the input text data.
        """
        super().__init__()
        if language == 'en':
            pretrain = "cardiffnlp/twitter-roberta-base-emotion"
        elif language == 'sp':
            pretrain = 'bertin-project/bertin-roberta-base-spanish'
        elif language == 'gr':
            pretrain = 'macedonizer/gr-roberta-base'
        else:
            raise ValueError('Language not recognized! Only avalaible languages are gr, sp and en!')

        self.model = XLMRobertaForSequenceClassification.from_pretrained(pretrain, num_labels=num_labels,
                                                                         ignore_mismatched_sizes=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the XLM-RoBERTa model.

        Args:
            input_ids (torch.Tensor): The input token ids.
            attention_mask (torch.Tensor): The attention mask for the input.
            labels (torch.Tensor, optional): The ground truth labels for the input. Defaults to None.

        Returns:
            torch.Tensor or tuple: The output logits or a tuple containing the logits and loss.
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.

        Args:
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        input_ids, attention_mask, labels = batch[0], batch[1], batch[3]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        #self.log("train_loss", loss, prog_bar=True)
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
        input_ids, attention_mask, labels = batch[0], batch[1], batch[3]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        #self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Testing step for PyTorch Lightning.

        Args:
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the accuracy and F1 score for the current batch.
        """
        input_ids, attention_mask, gold_labels, labels = batch[0], batch[1], batch[2], batch[3]
        outputs = self(input_ids, attention_mask, labels)
        predicted = torch.argmax(outputs.logits, 1)
        loss = outputs.loss
        results = {}
        results['accuracy'] = accuracy_score(gold_labels.cpu(), predicted.cpu())
        results['f1'] = f1_score(gold_labels.cpu(), predicted.cpu(), average='macro')
        return results

    def configure_optimizers(self):
        """
        Configure the optimizer for PyTorch Lightning.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        return torch.optim.AdamW(self.model.parameters(), lr=2e-5)