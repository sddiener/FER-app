import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from src import lib


def predict_emotion(image_path: str, checkpoint_path: str, device: str):
    # Load the image
    image = Image.open(image_path).convert("L")
    image = np.array(image)

    # Load the model
    model = lib.LitCNN.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)

    # Convert the image to a tensor
    image_tensor = ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    prediction = model(image_tensor)  # Predict the emotion
    prediction = prediction.cpu().detach().numpy()
    return prediction


if __name__ == "__main__":
    args = lib.parse_prediction_args(lib.DEFAULT_PREDICTION_ARGS)
    prediction = predict_emotion(
        args.image_path, args.checkpoint_path, args.device)
    print(f"Predicted emotion: {prediction}")
