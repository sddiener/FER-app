import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from collections import Counter

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
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

# custom packages
from src.utils import load_data
from src.mymodelclasses import ResNetCNN

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


def load_ferplus_dataset(data_file, image_dir):
    # Load the data file into a pandas dataframe
    df = pd.read_csv(data_file)

    # Extract the emotion labels as a 2D array
    labels = df.iloc[:, 2:10].values

    # Convert the labels to one-hot encoding
    num_classes = 8
    labels_one_hot = np.eye(num_classes)[labels]

    # Extract the image paths and usage as lists
    image_paths = df.iloc[:, 1].values
    usage = df.iloc[:, 0].values

    # Load the images from disk and store in a numpy array
    images = []
    for path in image_paths:
        img = cv2.imread(image_dir + "/" + path, 0)
        img = cv2.resize(img, (48, 48))
        images.append(img)
    images = np.array(images)

    # Return the images and labels as a tuple
    return images, labels_one_hot, usage
