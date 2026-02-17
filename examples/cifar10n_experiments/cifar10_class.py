from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class CIFAR10CustomClass(Dataset):
    """
    Simple dataset class for the CIFAR-10 dataset.

    Args:
        data (numpy.ndarray): Input image data as a NumPy array.
        labels (numpy.ndarray): Labels corresponding to the input data.
        transform (torchvision.transforms.Compose, optional): Transforms to be applied to the input data.

    Attributes:
        data (numpy.ndarray): Input image data as a NumPy array.
        labels (numpy.ndarray): Labels corresponding to the input data.
        transform (torchvision.transforms.Compose): Transforms to be applied to the input data.
    """

    def __init__(
        self,
        data,
        labels,
        transform,
    ):
        super(CIFAR10CustomClass, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Retrieves a single data sample and its corresponding label from the dataset.

        Args:
            index (int): Index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding label.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
