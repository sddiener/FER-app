import os
from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from src.lib import LitCNN, parse_prediction_args


def predict_emotion(image_path: str, model_path: str, device: str):
    # Load the image
    image = Image.open(image_path).convert("L")
    image = np.array(image)

    # Load the model
    model = LitCNN.load_from_checkpoint(model_path)
    model.freeze()
    model = model.to(device)

    # Convert the image to a tensor
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Predict the emotion
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.cpu().numpy()[0]

    return predicted


if __name__ == "__main__":
    # declare manual args list to pass to parse_args()
    FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data"
    MODEL_DIR = "C:/Users/stefan/Github/FER-app/results/models"

    # define default values for the arguments
    default_args = {
        'image_path': f"{FER_DATA_DIR}/FER2013Test/fer0032222.png",
        'model_path': f"{MODEL_DIR}/ferplus_cnn_e2e_v0.pt",
        'device': 'cuda',
    }
    args = parse_prediction_args(default_args)
    predict_emotion(args.image_path, args.model_path, args.device)

    # python prediction_module.py --image_path C:/Users/stefan/Github/FER-app/data/ferplus/data/FER2013Test/fer0032222.png --model_path C:/Users/stefan/Github/FER-app/results/models/ferplus_cnn_e2e_v0.pt
