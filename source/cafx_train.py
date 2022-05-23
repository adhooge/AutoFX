import pathlib

from source.models.CAFx import CAFx
from source.data.datamodules import FeaturesDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import pedalboard as pdb

CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/modulation_guitar_mono_cut")
OUT_OF_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050_cut")
NUM_EPOCHS = 200

PARAM_RANGE = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
CHECKPOINT = pathlib.Path("/home/alexandre/logs/cafx/version_1/checkpoints/epoch=99-step=93600.ckpt")
logger = TensorBoardLogger("/home/alexandre/logs", name="cafx")

datamodule = FeaturesDataModule(CLEAN_PATH, PROCESSED_PATH, OUT_OF_DOMAIN_PATH, seed=2)


cafx = CAFx([pdb.Chorus], 1, PARAM_RANGE, 28, loss_stamps=[50, 200])

checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)

trainer = pl.Trainer(gpus=2, logger=logger, max_epochs=NUM_EPOCHS,
                     strategy= 'ddp', auto_select_gpus=True,
                     callbacks=[checkpoint_callback], log_every_n_steps=100, detect_anomaly=True)
trainer.fit(cafx, datamodule=datamodule,
            ckpt_path=CHECKPOINT)
