from loguru import logger
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.lib import parse_training_args, FERPlusDataModule, LitCNN


def train_model(train_data_dir, val_data_dir, test_data_dir, model_save_path, batch_size=32, epochs=10,
                num_dl_workers=4, device='cuda', debug=False):

    logger.info(f"Training with arguments: train_data_dir={train_data_dir}, val_data_dir={val_data_dir}, "
                f"test_data_dir={test_data_dir}, model_save_path={model_save_path}, batch_size={batch_size}, "
                f"epochs={epochs}, num_dl_workers={num_dl_workers}, device={device}, debug={debug}")

    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    # Instantiate the FERPlusDataModule
    data_module = FERPlusDataModule(train_data_dir, val_data_dir, test_data_dir, batch_size, num_dl_workers, debug)

    # Instantiate the model
    model = LitCNN(input_shape=(batch_size, 1, 48, 48),  num_classes=9, lr=0.00001)
    device = torch.device(device)
    model = model.to(device)

    # Instantiate the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=epochs,
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
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    # declare manual args list to pass to parse_args()
    FER_DATA_DIR = "C:/Users/stefan/Github/FER-app/data/ferplus/data"

    # define default values for the arguments
    default_args = {
        'train_data_dir': f"{FER_DATA_DIR}/FER2013Train",
        'val_data_dir': f"{FER_DATA_DIR}/FER2013Valid",
        'test_data_dir': f"{FER_DATA_DIR}/FER2013Test",
        'model_save_path': 'C:/Users/stefan/Github/FER-app/results/models/ferplus_cnn_e2e_v1.pt',
        'batch_size': 256,
        'epochs': 3,
        'num_dl_workers': 0,
        'device': 'cuda',
        'debug': False
    }
    args = parse_training_args(default_args)
    train_model(args.train_data_dir, args.val_data_dir, args.test_data_dir, args.model_save_path,
                args.batch_size, args.epochs, args.num_dl_workers, args.device, args.debug)
