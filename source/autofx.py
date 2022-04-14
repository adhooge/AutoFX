"""
Convolutional Neural Network for parameter estimation
"""

import os
import pathlib
import pedalboard as pdb
from typing import Any, Optional

import torchaudio.transforms
from carbontracker.tracker import CarbonTracker
import auraloss
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
import sys
from multiband_fx import MultiBandFX

sys.path.append('..')


class AutoFX(pl.LightningModule):
    def _shape_after_conv(self, x):
        """
        Return shape after Conv2D according to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        :param x:
        :return:
        """
        batch_size, c_in, h_in, w_in = x.shape
        for conv in self.conv:
            h_out = torch.floor((h_in + 2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0] - 1) - 1)/conv.stride[0] + 1)
            w_out = torch.floor((w_in + 2*conv.padding[1] - conv.dilation[1]*(conv.kernel_size[1] - 1) - 1)/conv.stride[1] + 1)
            h_in = h_out
            w_in = w_out
        c_out = self.conv[-1].out_channels
        return batch_size, c_out, h_out, w_out

    def __init__(self, fx: pdb.Plugin, num_bands: int, tracker: bool = False, rate: int = 16000,
                 conv_ch: list[int] = [6, 16, 32], conv_k: list[int] = [5, 5, 5],
                 fft_size: int = 1024, hop_size: int = 256, audiologs: int = 8,
                 mrstft_fft: list[int] = [64, 128, 256, 512, 1024, 2048],
                 mrstft_hop: list[int] = [16, 32, 64, 128, 256, 512]):
        # TODO: Make attributes changeable from arguments
        super().__init__()
        self.conv = []
        for c in range(len(conv_ch)):
            self.conv.append(nn.Sequential(nn.Conv2d(1, conv_ch[c], conv_k[c]), nn.BatchNorm2d(conv_ch[c]), nn.ReLU()))
        self.gru = nn.GRU(16032, 512, batch_first=True)
        self.fcl = nn.Linear(512, 8)
        self.activation = nn.Sigmoid()
        self.loss = nn.L1Loss()
        self.tracker_flag = tracker
        self.tracker = None
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(mrstft_fft,
                                                            mrstft_hop,
                                                            mrstft_fft,
                                                            device="cpu")       # TODO: Manage device properly
        self.num_bands = num_bands
        self.mbfx = MultiBandFX(fx, num_bands)
        self.rate = rate
        self.spectro = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=hop_size, power=2)
        self.inv_spectro = torchaudio.transforms.InverseSpectrogram(n_fft=fft_size, hop_length=hop_size)
        self.audiologs = audiologs

    def forward(self, x, *args, **kwargs) -> Any:
        x = self.spectro(x)
        for conv in self.conv:
            x = conv(x)
        batch_size, channels, h_out, w_out = x.shape
        x = x.view(batch_size, w_out, channels, h_out)
        x = x.view(batch_size, w_out, -1)
        x, _ = self.gru(x, torch.zeros(1, batch_size, 512, device=x.device))
        x = self.fcl(x)
        x = torch.mean(x, 1)
        x = self.activation(x)
        return x

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        data, label = batch
        batch_size = data.shape[0]
        pred = self.forward(data)
        rec = torch.zeros(batch_size, 32000)
        for (i, snd) in enumerate(data):
            for b in range(self.num_bands):
                # TODO: Make it fx agnostic
                self.mbfx.mbfx[b][0].drive_db = pred[i][b]
                self.mbfx.mbfx[b][1].gain_db = pred[i][self.num_bands + b]
            rec[i] = self.mbfx(snd.to(torch.device('cpu')), self.rate)
        self.log("Train: Spectral loss",
                 self.mrstft(rec.to(torch.device('cpu')), data.to(torch.device('cpu'))))  # TODO: fix device management
        loss = self.loss(pred, label)
        self.log("train_loss", loss)
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            self.log("Train: Param {} distance".format(i), val)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        data, label = batch
        batch_size = data.shape[0]
        pred = self.forward(data)
        rec = torch.zeros(batch_size, 32000)  # TODO: fix hardcoded value
        for (i, snd) in enumerate(data):
            for b in range(self.num_bands):
                # TODO: Make it fx agnostic
                self.mbfx.mbfx[b][0].drive_db = pred[i][b]
                self.mbfx.mbfx[b][1].gain_db = pred[i][self.num_bands + b]
            rec[i] = self.mbfx(snd.to(torch.device('cpu')),
                               self.rate)  # TODO: Je réapplique l'effet sur l'audio déjà avec effet donc forcément ça se ressemble
        self.log("Test: Spectral loss",
                 self.mrstft(rec.to(torch.device("cpu")), data.to(torch.device("cpu"))))  # TODO: Fix device management
        loss = self.loss(pred, label)
        self.log("validation_loss", loss)
        for l in range(self.audiologs):
            self.logger.experiment.add_audio(f"Audio/{l}/Original", data[l],
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Matched", rec[l],
                                             sample_rate=self.rate, global_step=self.global_step)
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            self.log("Test: Param {} distance".format(i), val)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.tracker_flag and self.tracker is None:
            self.tracker = CarbonTracker(epochs=400, epochs_before_pred=10, monitor_epochs=10,
                                         log_dir=self.logger.log_dir, verbose=2)
        if self.tracker_flag:
            self.tracker.epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.tracker_flag:
            self.tracker.epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
