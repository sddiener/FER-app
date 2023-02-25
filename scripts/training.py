# This file is the main file for model training and testing
import os
from typing import Tuple
import argparse
from loguru import logger
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from torchvision import transforms

import src.lib as lib


class FERPlusDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),  # resize to 224x224
            # stack grayscale image along channels dimension
            transforms.Lambda(lambda x: torch.stack([x[0], x[0], x[0]], dim=0)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transform(image)
        label = self.labels[index]
        return image, label


def load_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = np.array(img)
    return img


def load_ferplus_dataset(train_folder):
    """
    Loads the FERPlus dataset from the specified train, val, or test folder.
    Source for data: https://github.com/microsoft/FERPlus
    """
    images = []
    labels = []

    labels_file = os.path.join(train_folder, "label.csv")

    labels_df = pd.read_csv(labels_file, header=None)
    labels_df.columns = ["image_names", "dim", "neutral", "happiness", "surprise", "sadness", "anger", "disgust",
                         "anger", "disgust", "fear", "contempt"]

    # Iterate over the lines in the labels CSV file for this train split
    for __, row in labels_df.iterrows():
        # Get the image path and label
        image_path = os.path.join(train_folder, row["image_names"])
        image = load_image(image_path)
        label = row[2:-1].values  # use probabilities

        images.append(image)
        labels.append(label)

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels, dtype=np.float32) / 10.0  # normalize probabilities

    return images, labels


class FERPlusDataModule(LightningDataModule):
    def __init__(self, ferplus_data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_folder = ferplus_data_dir

    def setup(self, stage=None):
        if stage == "fit" or stage is None:  # Assign train/val datasets for use in dataloaders
            train_images, train_labels = load_ferplus_dataset(os.path.join(self.data_folder, "FER2013Train"))
            val_images, val_labels = load_ferplus_dataset(os.path.join(self.data_folder, "FER2013Valid"))

            self.train_dataset = FERPlusDataset(train_images, train_labels)
            self.val_dataset = FERPlusDataset(val_images, val_labels)

        if stage == "test" or stage is None:  # Assign test dataset for use in dataloader(s)
            test_images, test_labels = load_ferplus_dataset(os.path.join(self.data_folder, "FER2013Test"))
            self.test_dataset = FERPlusDataset(test_images, test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


def main(args_list=None):
    args = parse_args(args_list)
    logger.info(f"Training with arguments: {args}")

    # Instantiate the FERPlusDataModule

    data_module = FERPlusDataModule(
        ferplus_data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    # test for debugging
    data_module.setup()
    data_module.train_dataset[0]

    # Instantiate the model
    model = MyModel()

    # Instantiate the trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        logger=TensorBoardLogger("logs/", name=args.exp_name),
        checkpoint_callback=ModelCheckpoint(
            dirpath="checkpoints/",
            filename="best-checkpoint",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    test_result = trainer.test(model, datamodule=data_module)

    # Log test result
    logger.info(f"Test results: {test_result[0]}")

    # Save the trained model
    torch.save(model.state_dict(), args.save_path)


def parse_args(custom_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="C:\\Users\\stefan\\Github\\FER-app\\data\\ferplus\\data",
                        help="path to ferplus data directory",)
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
    custom_args = ["--data_dir", "C:\\Users\\stefan\\Github\\FER-app\\data\\ferplus\\data",
                   "--batch_size", "32"]
    main(custom_args)
