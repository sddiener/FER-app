from loguru import logger
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from src import lib


def train_model(train_data_dir, val_data_dir, test_data_dir, ckpt_dir, model_name, batch_size=32, epochs=10,
                num_dl_workers=4, device='cuda', debug=False):

    logger.info(f"Training with arguments:\n"
                f"  train_data_dir = {train_data_dir}\n"
                f"  val_data_dir   = {val_data_dir}\n"
                f"  test_data_dir  = {test_data_dir}\n"
                f"  ckpt_dir       = {ckpt_dir}\n"
                f"  batch_size     = {batch_size}\n"
                f"  epochs         = {epochs}\n"
                f"  num_dl_workers = {num_dl_workers}\n"
                f"  device         = {device}\n"
                f"  debug          = {debug}")

    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    # Instantiate the FERPlusDataModule
    logger.info(f"Creating FERPlusDataModule ...")
    data_module = lib.FERPlusDataModule(train_data_dir, val_data_dir, test_data_dir,
                                    batch_size, num_dl_workers, debug)

    # Instantiate the model
    logger.info(f"Creating LitCNN ...")
    model = lib.LitCNN(input_shape=(batch_size, 1, 48, 48),  num_classes=9, lr=0.00001)
    device = torch.device(device)
    model = model.to(device)
    logger.info(f"GPU usage: {torch.cuda.memory_allocated(device) / 1e9} GB")

    # Instantiate the trainer
    logger.info(f"Creating Trainer ...")
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=epochs,
        logger=TensorBoardLogger("logs/", name="ferplus"),
        log_every_n_steps=5,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20),
            ModelCheckpoint(
                dirpath=f"{ckpt_dir}",
                filename=f"{model_name}",
                save_top_k=1,
                monitor="val_loss",
                mode="min"
            )
        ]
    )

    # Train the model
    trainer.fit(model, data_module)
    # Checkpoint save path
    logger.info(f"Saving model to {ckpt_dir} ...")
    # Test the model
    test_result = trainer.test(model, datamodule=data_module)
    logger.info(f"Test results: {test_result[0]}")


if __name__ == "__main__":
    args = lib.parse_training_args(lib.DEFAULT_TRAINING_ARGS)
    train_model(args.train_data_dir, args.val_data_dir, args.test_data_dir, args.ckpt_dir, args.model_name,
                args.batch_size, args.epochs, args.num_dl_workers, args.device, args.debug)
