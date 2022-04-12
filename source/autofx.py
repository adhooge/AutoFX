"""
Convolutional Neural Network for parameter estimation
"""

import os
import pathlib
from typing import Any, Optional

from carbontracker.tracker import CarbonTracker

import numpy as np
import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import sys

import util

sys.path.append('..')



class AutoFX(pl.LightningModule):
    def __init__(self, tracker: CarbonTracker = None):
        # TODO: Make attributes changeable from arguments
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5), nn.BatchNorm2d(6), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.BatchNorm2d(16), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 5), nn.BatchNorm2d(32), nn.ReLU())
        self.gru = nn.GRU(16032, 512, batch_first=True)
        self.fcl = nn.Linear(512, 8)
        self.activation = nn.Sigmoid()
        self.loss = nn.L1Loss()
        self.tracker = tracker

    def forward(self, x, *args, **kwargs) -> Any:
        batch_size, audio_channels, fft_size, num_frames = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, 114, 32, 501)        # TODO: Remove hardcoded values
        x = x.view(batch_size, 114, -1)
        x, _ = self.gru(x, torch.zeros(1, batch_size, 512, device=x.device))
        x = self.fcl(x)
        x = torch.mean(x, 1)
        x = self.activation(x)
        return x

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        data, label = batch
        pred = self.forward(data)
        loss = self.loss(pred, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        data, label = batch
        pred = self.forward(data)
        loss = self.loss(pred, label)
        self.log("validation_loss", loss)
        return loss

    def on_train_start(self) -> None:
        dataiter = iter(self.trainer.val_dataloaders[0])
        spectro, labels = dataiter.next()
        self.logger.experiment.add_images("Input spectrograms", spectro)

    def on_train_epoch_start(self) -> None:
        if self.tracker is not None:
            self.tracker.epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.tracker is not None:
            self.tracker.epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
