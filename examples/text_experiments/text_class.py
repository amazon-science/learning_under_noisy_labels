import torch
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

from src.iaa_api import InterAnnotatorAgreementAPI

class PersTecData(torch.utils.data.Dataset):
    """
    PyTorch dataset class for the PersTec dataset.

    This class loads the PersTec dataset from a CSV file, preprocesses the data, and provides methods
    for accessing the data during training or evaluation.
    """

    def __init__(self, language: str = "en", aggregation_type: str = "posterior"):
        """
        Initialize the PersTecData dataset.

        Args:
            language (str, optional): The language of the dataset. Defaults to "en".
            aggregation_type (str, optional): The type of annotation aggregation to use. Defaults to "posterior".
        """
        # Load the dataset from a CSV file
        dataset = pd.read_csv(
            f"hf://datasets/rbnuria/SentiMP-{language.capitalize()}/{language}.csv"
        )
        # Remove rows with missing values in the 'label_2' column
        dataset = dataset.dropna(subset=["label_2"], how="any")

        # Store the full text and language
        self.text = dataset["full_text"]
        self.language = language

        # Select the appropriate tokenizer based on the language
        if self.language == "en":
            tokenizer = AutoTokenizer.from_pretrained(
                "cardiffnlp/twitter-roberta-base-emotion",
                ignore_mismatched_sizes=True,
                use_fast=False,
            )
        elif self.language == "gr":
            tokenizer = AutoTokenizer.from_pretrained(
                "macedonizer/gr-roberta-base",
                ignore_mismatched_sizes=True,
                use_fast=False,
            )
        elif self.language == "sp":
            tokenizer = AutoTokenizer.from_pretrained(
                "bertin-project/bertin-roberta-base-spanish",
                ignore_mismatched_sizes=True,
                use_fast=False,
            )
        else:
            raise ValueError(
                "Language not recognized! Only avalaible languages are gr, sp and en!"
            )

        # Prepare annotations and labels
        annotations = (
            np.array(
                dataset[["label_1", "label_2", "label_3"]].values.tolist(), dtype=int
            )
            + 1
        )
        labels = [1 + a for a in dataset["gold_label"].tolist()]
        self.y = torch.tensor(labels)

        # Use InterAnnotatorAgreementAPI to aggregate annotations
        api = InterAnnotatorAgreementAPI(annotations=annotations)
        self.aggregated_annotations = api.return_annotations(
            aggregation_type, return_torch=True
        )

        # Tokenize the text if a tokenizer is provided
        self.tokenized = False
        if tokenizer != None:
            self.tokenized = True
            tokenized = tokenizer(
                self.text.tolist(),
                padding="max_length",
                truncation=True,
                max_length=128,
            )
            self.input_ids = torch.tensor(tokenized["input_ids"])
            self.attention_mask = torch.tensor(tokenized["attention_mask"])

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input data and labels for the sample.
                   If the data is tokenized, the tuple contains (input_ids, attention_mask, label, aggregated_label).
                   If the data is not tokenized, the tuple contains (text, label, aggregated_label).
        """
        if self.tokenized:
            # Return tokenized input, attention mask, label, and aggregated label
            sample = self.input_ids[index]
            mask = self.attention_mask[index]
            label = self.y[index]
            aggregated_label = self.aggregated_annotations[index]
            return sample, mask, label, aggregated_label
        else:
            # Return raw text, label, and aggregated label
            sample = self.text[index]
            label = np.squeeze(self.y[index])
            aggregated_label = self.aggregated_annotations[index]
            return sample, label, aggregated_label

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.tokenized:
            return self.input_ids.shape[0]
        else:
            return self.text.shape[0]