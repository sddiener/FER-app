# This file is the main file for model training and testing
import os
from typing import Tuple
import argparse
from loguru import logger
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import src.lib as lib
from src.lib import FERPlusDataModule, LitCNN


def main(args_list=None):
    args = lib.parse_training_args(args_list)
    logger.info(f"Training with arguments: {args}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    # Instantiate the FERPlusDataModule
    data_module = FERPlusDataModule(args.train_data_dir, args.val_data_dir,
                                    args.test_data_dir, args.batch_size, args.num_dl_workers, args.debug)

    # Instantiate the model
    model = LitCNN(input_shape=(args.batch_size, 1, 48, 48),  num_classes=9, lr=0.00001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Instantiate the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=args.epochs,
        logger=TensorBoardLogger("logs/", name="ferplus"),
        log_every_n_steps=5,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20),
            ModelCheckpoint(
                dirpath="results/checkpoints/",
                filename="best-checkpoint",
                save_top_k=1,
                monitor="val_loss",
                mode="min"
            )
        ]
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    test_result = trainer.test(model, datamodule=data_module)

    # Log test result
    logger.info(f"Test results: {test_result[0]}")

    # Save the trained model
    torch.save(model.state_dict(), args.model_save_path)


if __name__ == "__main__":
    # declare manual args list to pass to parse_args()
    FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data"
    args_list = [
        "--train_data_dir",  f"{FER_DATA_DIR}/FER2013Train",
        "--val_data_dir",  f"{FER_DATA_DIR}/FER2013Valid",
        "--test_data_dir",  f"{FER_DATA_DIR}/FER2013Test",
        '--model_save_path', 'C:/Users/stefan/Github/FER-app/results/models/ferplus_cnn_e2e.pt',
        '--batch_size', '256',
        '--num_dl_workers', '0',
        '--epochs', '100',
        '--debug', 'False',
    ]
    main(args_list)
