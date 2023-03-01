import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from src import lib


def main(checkpoint_path: str, device: str, image: np.array = None, image_path: str = None):
    assert image is not None or image_path is not None, "Either image or image_path must be provided"

    if image_path is not None:
        image = Image.open(image_path)

    model = lib.load_model(checkpoint_path, device)
    image_tensor = lib.img_to_tensor(image)
    prediction = lib.predict_emotion(model, image_tensor, device)
    return prediction


if __name__ == "__main__":
    args = lib.parse_prediction_args(lib.DEFAULT_PREDICTION_ARGS)
    prediction = main(args.image_path, args.checkpoint_path, args.device)
    print(f"Predicted emotion: {prediction}")
