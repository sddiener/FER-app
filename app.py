import streamlit as st
from src import prediction_module, lib


def main():
    model = lib.load_model(lib.LATEST_CHECKPOINT_PATH, device='cpu')

    st.title("Facial Emotion Recognition")

    uploaded_file = st.file_uploader("Upload an image of a face", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        prediction = lib.predict_emotion(model, uploaded_file)
        st.write(prediction)

        # TODO - Display the top 3 emotions in nicer format.
        # st.write(prediction)
        # st.write(type(prediction))
        # for i, p in enumerate(prediction):
        #     st.write(f"{lib.LABEL_DICT[i]}: {p}")
        # # prediction = lib.get_top_n_emotions(prediction, n=3)
        # # Create a list of tuples containing each emotion label and its probability
        # emotions = [(lib.LABEL_DICT[i], p) for i, p in enumerate(prediction)]
        # # Sort the list of emotions by probability in descending order
        # st.write("emotions")
        # st.write(type(emotions))
        # st.write(len(emotions))
        # st.write(emotions)
        # sorted_emotions = sorted(emotions, key=lambda x: x[1], reverse=True)
        # st.write("sorted_emotions")
        # st.write(sorted_emotions)


if __name__ == '__main__':
    main()
