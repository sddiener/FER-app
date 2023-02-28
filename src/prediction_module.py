import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from src import lib


def main(image_path: str, checkpoint_path: str, device: str):
    image = lib.load_image(image_path)
    model = lib.load_model(checkpoint_path, device)
    prediction = lib.predict_emotion(image, model)
    return prediction


if __name__ == "__main__":
    args = lib.parse_prediction_args(lib.DEFAULT_PREDICTION_ARGS)
    prediction = main(
        args.image_path, args.checkpoint_path, args.device)
    print(f"Predicted emotion: {prediction}")
