import pytest
import os
from src.lib import FERPlusDataModule
from src import training_main
from src import lib

# **********************************************************************************************************************
# Global Arguments
# **********************************************************************************************************************
FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data"
REQUIRED_ARGS = [
    "--train_data_dir",  f"{FER_DATA_DIR}/FER2013Train",
    "--val_data_dir",  f"{FER_DATA_DIR}/FER2013Valid",
    "--test_data_dir",  f"{FER_DATA_DIR}/FER2013Test",
    '--model_save_path', 'C:/Users/stefan/Github/FER-app/results/models//ferplus_cnn_e2e.pt'
]


# **********************************************************************************************************************
# Test Data & DataModule
# **********************************************************************************************************************
def test_data_exists():
    # Arguments
    args_list = REQUIRED_ARGS + ['--debug', 'True']
    args = lib.parse_training_args(args_list)

    # Check that the data exists
    assert os.path.exists(args.train_data_dir)
    assert os.path.exists(args.val_data_dir)
    assert os.path.exists(args.test_data_dir)

    # Check that the data is not empty
    assert len(os.listdir(args.train_data_dir)) > 0
    assert len(os.listdir(args.val_data_dir)) > 0
    assert len(os.listdir(args.test_data_dir)) > 0


def test_data_module_creation():
    # Arguments
    args_list = REQUIRED_ARGS + ['--debug', 'True']
    args = lib.parse_training_args(args_list)

    # Create FERPlusDataModule
    data_module = FERPlusDataModule(args.train_data_dir, args.val_data_dir, args.test_data_dir, args.batch_size)

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


def test_training_main():
    # Arguments
    args_list = REQUIRED_ARGS + ['--debug', 'True'] + ['--epochs', '1']
    args = lib.parse_training_args(args_list)

    # Run training_main.py
    training_main.main(args_list)

    # Check that the model was saved
    assert os.path.exists(args.model_save_path)

    # Check that the model is not empty
    assert os.path.getsize(args.model_save_path) > 0
