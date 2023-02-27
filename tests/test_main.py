import pytest
import os
from src.lib import FERPlusDataModule
from src.training_module import train_model
from src.prediction_module import predict_emotion
from src.lib import parse_training_args

# **********************************************************************************************************************
# Global Arguments
# **********************************************************************************************************************
FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data"
DEFAULT_TRAINING_ARGS = {
    "train_data_dir": f"{FER_DATA_DIR}/FER2013Train",
    "val_data_dir": f"{FER_DATA_DIR}/FER2013Valid",
    "test_data_dir": f"{FER_DATA_DIR}/FER2013Test",
    "model_save_path": "C:/Users/stefan/Github/FER-app/results/models//ferplus_cnn_e2e.pt",
    "debug": True,
    "batch_size": 64,
    "num_dl_workers": 0,
    "epochs": 1
}


# **********************************************************************************************************************
# Test Data & DataModule
# **********************************************************************************************************************
def test_data_exists():
    # Check that the data exists
    assert os.path.exists(DEFAULT_TRAINING_ARGS["train_data_dir"])
    assert os.path.exists(DEFAULT_TRAINING_ARGS["val_data_dir"])
    assert os.path.exists(DEFAULT_TRAINING_ARGS["test_data_dir"])

    # Check that the data is not empty
    assert len(os.listdir(DEFAULT_TRAINING_ARGS["train_data_dir"])) > 0
    assert len(os.listdir(DEFAULT_TRAINING_ARGS["val_data_dir"])) > 0
    assert len(os.listdir(DEFAULT_TRAINING_ARGS["test_data_dir"])) > 0


def test_data_module_creation():
    # Create FERPlusDataModule
    data_module = FERPlusDataModule(DEFAULT_TRAINING_ARGS["train_data_dir"], DEFAULT_TRAINING_ARGS["val_data_dir"],
                                    DEFAULT_TRAINING_ARGS["test_data_dir"], DEFAULT_TRAINING_ARGS["batch_size"],
                                    DEFAULT_TRAINING_ARGS["num_dl_workers"], DEFAULT_TRAINING_ARGS["debug"])

    # test for debugging
    data_module.setup()
    data_module.train_dataset[0]

    # Check that the data module was created correctly
    # assert dimensions of images are correct
    assert data_module.train_dataset[0][0].shape == (1, 48, 48)  # (3, 224, 224)
    assert data_module.val_dataset[0][0].shape == (1, 48, 48)
    assert data_module.test_dataset[0][0].shape == (1, 48, 48)

    # assert dimensions of labels are correct
    assert data_module.train_dataset[0][1].shape == (9,)
    assert data_module.val_dataset[0][1].shape == (9,)
    assert data_module.test_dataset[0][1].shape == (9,)


def test_training_module():
    # Run training_main.py
    train_model(**DEFAULT_TRAINING_ARGS)

    # Check that the model was saved
    assert os.path.exists(DEFAULT_TRAINING_ARGS["model_save_path"])

    # Check that the model is not empty
    assert os.path.getsize(DEFAULT_TRAINING_ARGS["model_save_path"]) > 0


def test_predict_emotion():
    FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data",
    MODEL_DIR = "C:/Users/stefan/Github/FER-app/results/models/",
    image_path =  f"{FER_DATA_DIR}/FER2013Test/fer0032222.png",
    model_path =  f"{MODEL_DIR}/ferplus_cnn_e2e_v0.pt",
    device = "cuda"

    # Expected prediction for the test image
    expected_prediction = 2

    # Call the predict_emotion function
    prediction = predict_emotion(image_path, model_path, device)

    # Assert that the predicted emotion is the expected one
    assert prediction == expected_prediction, f"Prediction {prediction} does not match expected value {expected_prediction}"
