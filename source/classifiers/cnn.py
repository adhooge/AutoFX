"""
Convolutional Neural Network classifier on spectrograms for comparison.
"""

import os
import pathlib
from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import sys
sys.path.append('..')
from spectrogram_dataset import SpectroDataset


DATASET_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_one_folder")


class CnnClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(1, 6, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(6, 16, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Flatten(1),
                                   nn.Linear(21056, 120),
                                   nn.ReLU(),
                                   nn.Linear(120, 84),
                                   nn.ReLU(),
                                   nn.Linear(84, 11))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        data, label = batch
        pred = self.model(data)
        loss = self.loss(pred, label)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


dataset = SpectroDataset(DATASET_PATH / 'labels.csv', DATASET_PATH, idmt=True, rate=44100)
train, test = random_split(dataset, [15000, 5592])

cnn = CnnClassifier()
trainer = pl.Trainer(gpus=1)
trainer.fit(cnn, DataLoader(train))
