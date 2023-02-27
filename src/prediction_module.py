import os
from typing import Tuple
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from pytorch_lightning import LightningModule, Trainer
from src.lib import LitCNN, parse_prediction_args, find_latest_checkpoint


def predict_emotion(image_path: str, checkpoint_path: str, device: str):
    # Load the image
    image = Image.open(image_path).convert("L")
    image = np.array(image)

    # Load the model
    model = LitCNN.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)

    # Convert the image to a tensor
    image_tensor = ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    prediction = model(image_tensor)  # Predict the emotion
    prediction = prediction.cpu().detach().numpy()
    # test_dataset = Dataset(test_tensor)
    # test_generator = torch.utils.data.DataLoader(test_dataset, **test_params)

    # mynet.eval()
    # batch = next(iter(test_generator))
    # with torch.no_grad():
    #     predictions_single_batch = mynet(**unpacked_batch)
    return prediction


if __name__ == "__main__":
    # declare manual args list to pass to parse_args()
    # Project root directory
    ROOT_DIR = "C:/Users/stefan/Github/FER-app"
    CHECKPOINT_NAME = find_latest_checkpoint(  # latest checkpoint name (e.g. ferplus_litcnn-v0.ckpt)
        dir_path=os.path.join(ROOT_DIR, "results/checkpoints"), 
        model_name="ferplus_litcnn"
        )

    # define default values for the arguments
    default_args = {
        'image_path': f"{ROOT_DIR}/data/ferplus/data/FER2013Test/fer0032222.png",
        'checkpoint_path': f"{ROOT_DIR}/results/checkpoints/{CHECKPOINT_NAME}",
        'device': 'cuda',
    }
    args = parse_prediction_args(default_args)
    prediction = predict_emotion(args.image_path, args.checkpoint_path, args.device)
    print(f"Predicted emotion: {prediction}")
    # python prediction_module.py --image_path C:/Users/stefan/Github/FER-app/data/ferplus/data/FER2013Test/fer0032222.png --model_path C:/Users/stefan/Github/FER-app/results/models/ferplus_cnn_e2e_v0.pt
