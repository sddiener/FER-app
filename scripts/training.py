## This file is the main file for model training and testing
import src.lib as lib
import argparse

import cv2
import pandas as pd
import numpy as np
import os
import numpy as np
from PIL import Image

import os
import numpy as np
from PIL import Image

from loguru import logger


# Load image from path using PIL
def load_image(path):
    return np.array(Image.open(path).convert("L"))


# Load labels from CSV file
def load_labels(path):
    labels = []
    with open(path, "r") as f:
        next(f)  # skip header row
        for line in f:
            label = [float(x) for x in line.strip().split(",")[2:-1]]  # Use probabilities
            label /= sum(label)  # Normalize probabilities to sum to 1
            labels.append(label)
    return np.array(labels)


def load_ferplus_dataset(data_folder):
    """
    Loads the FERPlus dataset from a folder.

    Args:
        data_folder (str): The folder containing the FERPlus dataset.

    Returns:
        A tuple (X, y) containing the input images and corresponding labels.
        X is a numpy array of shape (n_samples, height, width) containing the
        input images. y is a numpy array of shape (n_samples, n_labels)
        containing the corresponding labels.
    """
    images = []
    labels = []

    # Iterate over the train, test, and valid splits
    for suffix in ["Train", "Test", "Valid"]:
        split_folder = os.path.join(data_folder, "FER2013" + suffix)
        labels_file = os.path.join(split_folder, "label.csv")

        labels_df = pd.read_csv(labels_file, header=None)
        labels_df.columns = [
            "image_names",
            "dim",
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
            "unknown",
            "NF",
        ]

        # Iterate over the lines in the labels CSV file for this split
        for i, row in labels_df.iterrows():
            # Get the image path and label
            image_path = os.path.join(split_folder, row["image_names"])
            image = load_image(image_path)
            label = row[2:-1].values  # use probabilities

            images.append(image)
            labels.append(label)

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels, dtype=np.float32) / 10.0  # normalize probabilities

    return images, labels


def main(args_list=None):
    args = parse_args(args_list)
    logger.info(f"Training with arguments: {args}")
    images, labels = load_ferplus_dataset(args.data_dir)
    print("done")
    # train_loader, val_loader, test_loader = None
    # model = lib.get_model(args)
    # trainer = lib.get_trainer(args)
    # trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)

    # train_loader, val_loader, test_loader = load_data(args)
    # model = train_model(train_loader, val_loader, args)
    # test_loss, test_acc = test_model(model, test_loader, args)
    # save_model(model, args.model_type)


def parse_args(custom_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="C:\\Users\\stefan\\Github\\FER-app\\data\\ferplus\\data",
        help="path to ferplus data directory",
    )
    parser.add_argument("--model_type", type=str, default="resnet18", help="type of model to use")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="device to train on")

    # check if sys args or custom args are passed
    if custom_args:
        return parser.parse_args(custom_args)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    # declare manual args list to pass to parse_args()
    custom_args = ["--data_dir", "C:\\Users\\stefan\\Github\\FER-app\\data\\ferplus\\data"]
    main(custom_args)
