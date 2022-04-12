import pathlib

from carbontracker.tracker import CarbonTracker
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from autofx import AutoFX
from mbfx_dataset import MBFXDataset

DATASET_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto")
NUM_EPOCHS = 400


logger = TensorBoardLogger("lightning_logs")
tracker = CarbonTracker(epochs=400, epochs_before_pred=10, monitor_epochs=10, log_dir=logger.log_dir, verbose=2)
# tracker = None
dataset = MBFXDataset(DATASET_PATH / 'params.csv', DATASET_PATH, rate=44100)
train, test = random_split(dataset, [15000, 5016])

cnn = AutoFX(tracker=tracker)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=NUM_EPOCHS)
trainer.fit(cnn, DataLoader(train, batch_size=64, num_workers=4), DataLoader(test, batch_size=64, num_workers=4))
