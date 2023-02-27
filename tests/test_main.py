import pytest
import os
from src import lib
from src.lib import FERPlusDataModule
from src.training_module import train_model
from src.prediction_module import predict_emotion
from src.lib import parse_training_args, parse_prediction_args, find_latest_checkpoint

# **********************************************************************************************************************
# Global Arguments
# **********************************************************************************************************************
ROOT_DIR = "C:/Users/stefan/Github/FER-app"
FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data"
CHECKPOINT_NAME = find_latest_checkpoint(dir_path=os.path.join(ROOT_DIR, "results/checkpoints"), model_name="ferplus_litcnn")

DEFAULT_TRAINING_ARGS = {
    "train_data_dir": f"{FER_DATA_DIR}/FER2013Train",
    "val_data_dir": f"{FER_DATA_DIR}/FER2013Valid",
    "test_data_dir": f"{FER_DATA_DIR}/FER2013Test",
    'ckpt_dir': f"{ROOT_DIR}/results/checkpoints",
    'model_name': "ferplus_litcnn",
    "debug": True,
    "batch_size": 64,
    "num_dl_workers": 0,
    "epochs": 1
}
 
DEFAULT_PREDICTION_ARGS = {
    'image_path': f"{ROOT_DIR}/data/ferplus/data/FER2013Test/fer0032222.png",
    'checkpoint_path': f"{ROOT_DIR}/results/checkpoints/{CHECKPOINT_NAME}",
    "device": "cuda",
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
    train_model(**DEFAULT_TRAINING_ARGS)  # TODO Give testing name to the model

    assert os.path.exists(DEFAULT_PREDICTION_ARGS["checkpoint_path"])  # Check that the checkpoint file exists
    assert os.path.getsize(DEFAULT_PREDICTION_ARGS["checkpoint_path"]) > 0  # Check that the file is not empty


def test_predict_emotion():
    # define default values for the arguments
    prediction = predict_emotion(**DEFAULT_PREDICTION_ARGS)
    prediction = prediction.argmax()
    # Expected prediction for the test image
    expected_prediction = 8

    # Assert that the predicted emotion is the expected one
    assert prediction == expected_prediction, f"Prediction {prediction} does not match expected value {expected_prediction}"
