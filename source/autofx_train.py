import pathlib

from carbontracker.tracker import CarbonTracker
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from autofx_nogru import AutoFX
from mbfx_dataset import MBFXDataset
from idmt_dataset import IDMTDataset
import pedalboard as pdb

DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")
# PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto_guitar_mono_int")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_non-linear_22050")
NUM_EPOCHS = 400

logger = TensorBoardLogger("/home/alexandre/logs", name="testing")

# tracker = None
# dataset = MBFXDataset(PROCESSED_PATH / 'params.csv', DATASET_PATH, PROCESSED_PATH, rate=22050)
dataset = IDMTDataset(PROCESSED_PATH / "fx2clean.csv", DATASET_PATH, PROCESSED_PATH)
# train, test = random_split(dataset, [28000, 4720*2])
train, test = random_split(dataset, [2808, 936])

cnn = AutoFX(fx=[pdb.Distortion(0), pdb.Gain(0)], num_bands=4, total_num_bands=32, param_range=[(10, 60), (-10, 10)],
             tracker=True, out_of_domain=True, audiologs=6,
             conv_k=[5, 5, 5, 5, 5], conv_ch=[64, 64, 64, 64, 64], mel_spectro=True,
             conv_stride=[2, 2, 2, 2, 2], loss_stamps=[50, 200])

checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=NUM_EPOCHS, auto_lr_find=True, auto_select_gpus=True,
                     callbacks=[checkpoint_callback], log_every_n_steps=100, track_grad_norm=2, detect_anomaly=True)
trainer.fit(cnn, DataLoader(train, batch_size=64, num_workers=4, shuffle=True),
            DataLoader(test, batch_size=64, num_workers=4, shuffle=True))
# trainer = pl.Trainer()
# trainer.fit(cnn, DataLoader(train, batch_size=64, num_workers=4, shuffle=True),
#            DataLoader(test, batch_size=64, num_workers=4, shuffle=True),
#            ckpt_path="/home/alexandre/logs/Full_train/version_6/checkpoints/epoch=49-step=21900.ckpt")
