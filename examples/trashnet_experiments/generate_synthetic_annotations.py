import os
import csv
import numpy as np
import re

num_annotators = 5  # Number of annotators
num_classes = 6  # Number of classes
num_images = 2527  # Number of images
labels_to_ids = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5,
}  # Mapping of labels to IDs

def get_label_from_filename(filename):
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

T = np.array(
    [
        [0.7, 0.3, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.1, 0.6, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.2, 0.7, 0.0, 0.0],
        [0.0, 0.1, 0.1, 0.1, 0.6, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
    ]
)  # Noise transition matrix

all_files = []
img_dir = os.getcwd() + "/../../data/dataset-resized"
for folder in os.listdir(img_dir):
    if os.path.isdir(os.path.join(img_dir, folder)):
        for item in os.listdir(os.path.join(img_dir, folder)):
            all_files.append(item)

# Generate annotations using the noise transition matrix and write to CSV file
with open(os.path.join(img_dir, "annotations.csv"), "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "annotations"])
    for image_name in all_files:
        true_class = labels_to_ids[get_label_from_filename(image_name)]
        annotations_per_annotator = []
        for i in range(num_annotators):
            annotation_probabilities = T[true_class]
            single_annotation = np.random.choice(
                num_classes, p=annotation_probabilities
            )
            annotations_per_annotator.append(single_annotation)
            # writer.writerow([image_name, single_annotation])
        writer.writerow([image_name, annotations_per_annotator])