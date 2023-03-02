import argparse
from loguru import logger
import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
import torchmetrics
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import src.lib as lib

# **********************************************************************************************************************
# Utils
# **********************************************************************************************************************

# Get the latest checkpoint

import re


def find_latest_checkpoint(dir_path, model_name) -> str:
    checkpoints = [f for f in os.listdir(dir_path) if f.endswith('.ckpt')]
    checkpoints_v = [f for f in checkpoints if re.match(r'^{}-v\d+.ckpt$'.format(model_name), f)]
    if checkpoints_v:
        latest_checkpoint = max(checkpoints_v, key=lambda f: int(re.search(r'\d+', f).group()))
    else:
        latest_checkpoint = max(checkpoints)
    return latest_checkpoint

# **********************************************************************************************************************
# CONFIG VARIABLES
# **********************************************************************************************************************


ROOT_DIR = "C:/Users/stefan/Github/FER-app"
FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data"
BEST_CHECKPOINT_NAME = "ferplus_litcnn-v18.ckpt"
BEST_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "results/checkpoints", BEST_CHECKPOINT_NAME)
# LATEST_CHECKPOINT_NAME = find_latest_checkpoint(
#     dir_path=os.path.join(ROOT_DIR, "results/checkpoints"),
#     model_name="ferplus_litcnn"
# )

DEFAULT_TRAINING_ARGS = {
    "train_data_dir": f"{FER_DATA_DIR}/FER2013Train",
    "val_data_dir": f"{FER_DATA_DIR}/FER2013Valid",
    "test_data_dir": f"{FER_DATA_DIR}/FER2013Test",
    'ckpt_dir': f"{ROOT_DIR}/results/checkpoints",
    'model_name': "ferplus_litcnn",
    "debug": True,
    "batch_size": 64,
    "num_dl_workers": 0,
    "epochs": 1
}

DEFAULT_PREDICTION_ARGS = {
    'image_path': f"{ROOT_DIR}/data/ferplus/data/FER2013Test/fer0032222.png",
    'checkpoint_path': f"{ROOT_DIR}/results/checkpoints/{BEST_CHECKPOINT_NAME}",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

LABEL_DICT = {
    0: "neutral",
    1: "happiness",
    2: "surprise",
    3: "sadness",
    4: "anger",
    5: "disgust",
    6: "anger",
    7: "disgust",
    8: "fear",
    9: "contempt"
}


def get_top_n_emotions(predictions, n=None, label_dict=LABEL_DICT):
    # Create a list of tuples containing each emotion label and its probability
    emotions = [(label_dict[i], p) for i, p in enumerate(predictions)]
    # Sort the list of emotions by probability in descending order
    sorted_emotions = sorted(emotions, key=lambda x: x[1], reverse=True)
    # Return the top n emotions and their probabilities as a new dictionary
    return dict(sorted_emotions[:n])


# **********************************************************************************************************************
# Argument Parsing
# **********************************************************************************************************************


def str_to_bool(s: str) -> bool:
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError("Cannot convert string to bool")


def parse_args_with_error_handling(parser, args_dict):
    try:
        if args_dict is None:
            return parser.parse_args()
        else:
            return parser.parse_args(namespace=argparse.Namespace(**args_dict))

    except argparse.ArgumentError as e:
        print(f"Error: {e}")
        parser.print_usage()
        exit(1)


def parse_training_args(args_dict=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # add arguments for each parameter
    parser.add_argument('--train_data_dir', type=str, help='path to train data directory')
    parser.add_argument('--val_data_dir', type=str, help='path to val data directory')
    parser.add_argument('--test_data_dir', type=str, help='path to test data directory')
    parser.add_argument('--ckpt_dir', type=str, help='path to checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--num_dl_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'cpu', help='device to train on')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode')

    return parse_args_with_error_handling(parser, args_dict)


def parse_prediction_args(args_dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="path to the image to predict")
    parser.add_argument("--checkpoint_path", type=str, help="path to the checkpoint to use for prediction")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for prediction")

    return parse_args_with_error_handling(parser, args_dict)


# **********************************************************************************************************************
# FERPlus - Data Loading
# **********************************************************************************************************************


def load_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = np.array(img)
    return img


def load_ferplus_dataset(data_folder, debug=False):
    """
    Loads the FERPlus dataset from the specified train, val, or test folder.
    Source for data: https://github.com/microsoft/FERPlus
    """
    images = []
    labels = []

    labels_file = os.path.join(data_folder, "label.csv")

    labels_df = pd.read_csv(labels_file, header=None)
    labels_df.columns = ["image_names", "dim", "neutral", "happiness", "surprise", "sadness", "anger",
                         "disgust", "anger", "disgust", "fear", "contempt"]

    # Iterate over the lines in the labels CSV file for this train split
    for __, row in labels_df.iterrows():
        # Get the image path and label
        image_path = os.path.join(data_folder, row["image_names"])
        image = load_image(image_path)
        label = row[2:-1].values  # use probabilities

        images.append(image)
        labels.append(label)

        if debug and len(images) >= 100:
            break

    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels, dtype=np.float32) / 10.0  # normalize probabilities
    logger.info(f"Loaded {len(images)} images from {data_folder}, shape: {images.shape}")
    return images, labels


# **********************************************************************************************************************
# PyTorch Lightning - Dataset & DataModule
# **********************************************************************************************************************


class FERPlusDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # to grayscale
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),  # resize to 224x224
                # stack grayscale image along channels dimension
                # transforms.Lambda(lambda x: torch.stack([x[0], x[0], x[0]], dim=0)),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transform(image)
        label = self.labels[index]
        return image, label


class FERPlusDataModule(LightningDataModule):
    def __init__(self, train_data_dir: str, valdata_dir: str, test_data_dir: str, batch_size: int = 32,
                 num_dl_workers: int = 1, debug: bool = False):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.valdata_dir = valdata_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_dl_workers = num_dl_workers
        self.debug = debug

    def setup(self, stage=None):
        logger.info("Setting up data module...")
        if (stage == "fit" or stage is None):  # Assign train/val datasets for use in dataloaders
            train_images, train_labels = lib.load_ferplus_dataset(self.train_data_dir, debug=self.debug)
            val_images, val_labels = lib.load_ferplus_dataset(self.valdata_dir, debug=self.debug)

            self.train_dataset = FERPlusDataset(train_images, train_labels)
            self.val_dataset = FERPlusDataset(val_images, val_labels)

        if (stage == "test" or stage is None):  # Assign test dataset for use in dataloader(s)
            test_images, test_labels = lib.load_ferplus_dataset(self.test_data_dir, debug=self.debug)
            self.test_dataset = FERPlusDataset(test_images, test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                          num_workers=self.num_dl_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                          num_workers=self.num_dl_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                          num_workers=self.num_dl_workers)

# **********************************************************************************************************************
# PyTorch Lightning - Models
# **********************************************************************************************************************

# src: https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html


class LitCNN(LightningModule):
    def __init__(self, input_shape, num_classes, lr, batch_size=32):
        super().__init__()
        self.input_shape = input_shape  # use the second dimension of input shape as input size
        self.in_channels = input_shape[1]
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.example_input_array = torch.randn((batch_size,) + input_shape[1:])  # for logging (must be batch shape)
        self.save_hyperparameters()  # save hyperparameters for checkpointing

        # CNN block - image size 100*100
        self.CNN_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # FC block - image batch size: torch.Size([32, 128, 12, 12])
        self.FC_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 6 * 6, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        x = self.CNN_layers(x)
        x = self.FC_layers(x)
        x = F.softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))  # send input to GPU
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.accuracy(y_hat.argmax(dim=1), y.argmax(dim=1)),
                 prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))  # send input to GPU
        loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.accuracy(y_hat.argmax(dim=1), y.argmax(dim=1)),
                 prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)


# src: https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html
class ResNetCNN(LightningModule):
    def __init__(self, num_classes, lr):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, task="multiclass"
        )
        self.example_input_array = T.randn(
            64, 3, 100, 100
        )  # for logging (must be batch shape)
        self.save_hyperparameters()  # add hparams to checkpoints
        self.lr = lr

        # ResNet Backbone
        # src:
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/transfer_learning.html#example-imagenet-computer-vision
        backbone_tmp = models.resnet50(pretrained=True)
        num_filters = (backbone_tmp.fc.in_features)  # get feature number before removing layer
        layers = list(backbone_tmp.children())[:-1]
        self.backbone = nn.Sequential(*layers)

        # use the pretrained model to classify the 7 emotions
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)  # 1 to ignore batch dimension
        x = self.classifier(x)  # no activation. softmax handled by cross entropy loss
        return x

    def configure_optimizers(self):
        optimizer = T.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.accuracy(y_hat, y), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred

# **********************************************************************************************************************
# Streamlit
# **********************************************************************************************************************


def load_model(checkpoint_path: str, device: str) -> lib.LitCNN:
    model = lib.LitCNN.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    return model


def img_to_tensor(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        # transforms.Normalize((0.5,), (0.5,))  # TODO: Check if this is needed
    ])
    img = transform(img).unsqueeze(0)
    return img


def predict_emotion(model, image: torch.tensor, device=None):
    if device is None:
        device = model.device

    image = image.to(device)
    model = model.to(device)
    output = model(image)

    # TODO: Refactor to show top 3 emotions with probability. Or use a bar chart.
    # convert output probabilities to predicted class
    output = output.cpu().detach()  # send to CPU
    class_idx = output.numpy().argmax()
    emotion = lib.LABEL_DICT[class_idx]
    return emotion

# **********************************************************************************************************************
# Plotting
# **********************************************************************************************************************


def plot_images_with_predictions(image_paths: List[str], predictions: List[str]):
    num_images = len(image_paths)
    num_rows = num_images // 5 + 1 if num_images % 5 != 0 else num_images // 5
    fig, axs = plt.subplots(num_rows, 5, figsize=(15, 5 * num_rows))
    axs = axs.ravel()
    for i, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        image = Image.open(image_path)
        axs[i].imshow(image, cmap="gray")
        axs[i].set_title(f"Predicted: {prediction}")
        axs[i].axis("off")
    for i in range(num_images, num_rows * 5):
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()
