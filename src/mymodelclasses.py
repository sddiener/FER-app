import torch as T
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential
from torch.utils.data import DataLoader, TensorDataset, random_split

import torchmetrics

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# src: https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html
class LitCNN(LightningModule):
    def __init__(self, input_size, num_classes, lr):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.example_input_array = T.randn(64, 3, 100, 100)  # for logging (must be batch shape)
        self.lr = lr
        self.input_size = input_size

        # CNN block - image size 100*100
        self.CNN_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(in_features=128 * 12 * 12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        x = self.CNN_layers(x)
        # x = x.view(x.size(0), -1)  # flatten
        x = self.FC_layers(x)  # no activation and no softmax at the end
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


# src: https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html
class ResNetCNN(LightningModule):
    def __init__(self, num_classes, lr):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, task="multiclass")
        self.example_input_array = T.randn(64, 3, 100, 100)  # for logging (must be batch shape)
        self.save_hyperparameters()  # add hparams to checkpoints
        self.lr = lr

        # ResNet Backbone
        # src: https://pytorch-lightning.readthedocs.io/en/latest/advanced/transfer_learning.html#example-imagenet-computer-vision
        backbone_tmp = models.resnet50(pretrained=True)
        num_filters = backbone_tmp.fc.in_features  # get feature number before removing layer
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
