import pathlib

from carbontracker.tracker import CarbonTracker
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from autofx import AutoFX
from mbfx_dataset import MBFXDataset
import pedalboard as pdb

DATASET_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto_guitar_mono")
NUM_EPOCHS = 20


logger = TensorBoardLogger("lightning_logs")

# tracker = None
dataset = MBFXDataset(DATASET_PATH / 'params.csv', DATASET_PATH, rate=44100)
train, test = random_split(dataset, [14000, 4720])


cnn = AutoFX(fx=[pdb.Distortion, pdb.Gain], num_bands=4, tracker=False)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=NUM_EPOCHS)
trainer.fit(cnn, DataLoader(train, batch_size=64, num_workers=4), DataLoader(test, batch_size=64, num_workers=4))
