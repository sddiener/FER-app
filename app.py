import streamlit as st
from src import prediction_module, lib
from PIL import Image
import numpy as np


def main():
    st.title("Facial Emotion Recognition")

    uploaded_file = st.file_uploader("Upload an image of a face", type=["jpg", "jpeg", "png"])
    # convert to numpy array

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        image_array = np.array(Image.open(uploaded_file))
        st.write(f"RAW Prediction: {prediction}")

        prediction = prediction_module.main(
            image=image_array,
            checkpoint_path=lib.BEST_CHECKPOINT_PATH,
            device='cpu')

        st.write(f"Predicted emotion: {prediction}")


if __name__ == '__main__':
    main()
