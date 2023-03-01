import streamlit as st
from src import prediction_module, lib


def main():
    model = lib.load_model(lib.LATEST_CHECKPOINT_PATH, device='cpu')

    st.title("Facial Emotion Recognition")

    uploaded_file = st.file_uploader("Upload an image of a face", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        prediction = lib.predict_emotion(model, uploaded_file)
        st.write(f"Predicted emotion: {prediction}")


if __name__ == '__main__':
    main()
