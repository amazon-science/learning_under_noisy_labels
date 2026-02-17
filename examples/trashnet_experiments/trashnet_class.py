from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
import re
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import numpy as np

class TrashNetCustomClass(Dataset):
    """
    Custom PyTorch dataset class for the TrashNet dataset.

    This class loads the TrashNet dataset from a directory and provides methods for accessing
    the images, labels, and annotations during training or evaluation.
    """

    def __init__(self, img_dir, transform=None):
        """
        Initialize the TrashNetCustomClass dataset.

        Args:
            img_dir (str): The path to the directory containing the dataset.
            transform (callable, optional): Optional transform to be applied on the images.
        """
        super(TrashNetCustomClass, self).__init__()

        self.labels_to_ids = {
            "cardboard": 0,
            "glass": 1,
            "metal": 2,
            "paper": 3,
            "plastic": 4,
            "trash": 5,
        }
        self.transform = transform
        self.img_dir = img_dir
        self.annotation_file = pd.read_csv(
            os.path.join(self.img_dir, "annotations.csv")
        )
        self.all_files = []
        self.all_labels = []
        self.all_annotations = []
        i = 0
        for folder in os.listdir(self.img_dir):
            if os.path.isdir(os.path.join(self.img_dir, folder)):
                for item in os.listdir(os.path.join(self.img_dir, folder)):
                    self.all_files.append(item)
                    self.all_labels.append(
                        self.labels_to_ids[self.get_label_from_filename(item)]
                    )
                    row = self.annotation_file.loc[
                        self.annotation_file["image_name"] == item
                    ]
                    annotations_str = (
                        row["annotations"].values[0].strip('"[]').split(",")
                    )
                    annotations = [int(j) for j in annotations_str]
                    self.all_annotations.insert(i, annotations)
                    i += 1
        self.all_annotations = np.array(self.all_annotations)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.all_files)

    def get_label_from_filename(self, filename):
        """
        Extract the label from the filename.

        Args:
            filename (str): The filename of the image.

        Returns:
            str: The label extracted from the filename.
        """
        match = re.match(r"(.+?)(\d+)", filename)
        if match:
            label = match.group(1)
            return label
        else:
            return None

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor, label, and annotations for the sample.
        """
        image = read_image(
            os.path.join(
                self.img_dir,
                self.get_label_from_filename(self.all_files[idx]),
                self.all_files[idx],
            )
        )
        label = self.all_labels[idx]
        annotations = np.array(self.all_annotations[idx])
        if self.transform:
            image = Image.open(
                os.path.join(
                    self.img_dir,
                    self.get_label_from_filename(self.all_files[idx]),
                    self.all_files[idx],
                )
            )
            image = self.transform(image)
        return image, label, annotations