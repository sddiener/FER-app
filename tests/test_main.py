import pytest
import os
from src import lib
from src.lib import FERPlusDataModule
from src.training_module import train_model
from src.prediction_module import predict_emotion
from src.lib import parse_training_args, parse_prediction_args, find_latest_checkpoint

# TODO Automate running test before commit


# **********************************************************************************************************************
# Test Data & DataModule
# **********************************************************************************************************************
def test_data_exists():
    # Check that the data exists
    assert os.path.exists(lib.DEFAULT_TRAINING_ARGS["train_data_dir"])
    assert os.path.exists(lib.DEFAULT_TRAINING_ARGS["val_data_dir"])
    assert os.path.exists(lib.DEFAULT_TRAINING_ARGS["test_data_dir"])

    # Check that the data lib.ty
    assert len(os.listdir(lib.DEFAULT_TRAINING_ARGS["train_data_dir"])) > 0
    assert len(os.listdir(lib.DEFAULT_TRAINING_ARGS["val_data_dir"])) > 0
    assert len(os.listdir(lib.DEFAULT_TRAINING_ARGS["test_data_dir"])) > 0


def test_data_module_creation():
    # Create FERPlusDataModule
    data_module = FERPlusDataModule(lib.DEFAULT_TRAINING_ARGS["train_data_dir"], lib.DEFAULT_TRAINING_ARGS["val_data_dir"],
                                    lib.DEFAULT_TRAINING_ARGS["test_data_dir"], lib.DEFAULT_TRAINING_ARGS["batch_size"],
                                    lib.DEFAULT_TRAINING_ARGS["num_dl_workers"], lib.DEFAULT_TRAINING_ARGS["debug"])

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
    train_model(**lib.DEFAULT_TRAINING_ARGS)  # TODO Give testing name to the model

    assert os.path.exists(lib.DEFAULT_PREDICTION_ARGS["checkpoint_path"])  # Check that the checkpoint file exists
    assert os.path.getsize(lib.DEFAULT_PREDICTION_ARGS["checkpoint_path"]) > 0  # Check that the file is not empty


def test_predict_emotion():
    prediction = predict_emotion(**lib.DEFAULT_PREDICTION_ARGS)

    # Assert that the prediction is not empty
    assert prediction.shape == (1, 9)
