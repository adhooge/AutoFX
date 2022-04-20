"""
Convolutional Neural Network for parameter estimation
"""

import os
import pathlib
import pedalboard as pdb
from typing import Any, Optional, Tuple

import torchaudio.transforms
from carbontracker.tracker import CarbonTracker
import auraloss
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
import sys
from multiband_fx import MultiBandFX
from math import floor
sys.path.append('..')


class AutoFX(pl.LightningModule):
    def _shape_after_conv(self, x: torch.Tensor or Tuple):
        """
        Return shape after Conv2D according to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        :param x:
        :return:
        """
        if isinstance(x, torch.Tensor):
            batch_size, c_in, h_in, w_in = x.shape
        else:
            batch_size, c_in, h_in, w_in = x
        for seq in self.conv:
            conv = seq[0]
            h_out = floor((h_in + 2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0] - 1) - 1)/conv.stride[0] + 1)
            w_out = floor((w_in + 2*conv.padding[1] - conv.dilation[1]*(conv.kernel_size[1] - 1) - 1)/conv.stride[1] + 1)
            h_in = h_out
            w_in = w_out
        c_out = self.conv[-1][0].out_channels
        return batch_size, c_out, h_out, w_out

    def __init__(self, fx: pdb.Plugin, num_bands: int, tracker: bool = False, rate: int = 22050, file_size: int = 44100,
                 conv_ch: list[int] = [64, 64, 64], conv_k: list[int] = [5, 5, 5], conv_stride: list[int] = [2, 2, 2],
                 fft_size: int = 1024, hop_size: int = 256, audiologs: int = 4, loss_weights: list[float] = [1, 1],
                 mrstft_fft: list[int] = [64, 128, 256, 512, 1024, 2048],
                 mrstft_hop: list[int] = [16, 32, 64, 128, 256, 512],
                 learning_rate: int = 0.001,
                 spectro_power: int = 2, mel_spectro: bool = True, mel_num_bands: int = 128, device = torch.device('cuda')):        # TODO: change
        super().__init__()
        self.conv = nn.ModuleList([])
        self.mbfx = MultiBandFX(fx, num_bands, device=torch.device('cpu'))
        self.num_params = num_bands * len(self.mbfx.settings[0])
        for c in range(len(conv_ch)):
            if c == 0:
                self.conv.append(nn.Sequential(nn.Conv2d(1, conv_ch[c], conv_k[c],
                                                         padding=int(conv_k[c]/2), stride=conv_stride[c]),
                                               nn.BatchNorm2d(conv_ch[c]), nn.ReLU()))
            else:
                self.conv.append(nn.Sequential(nn.Conv2d(conv_ch[c-1], conv_ch[c], conv_k[c],
                                                         stride=conv_stride[c], padding=int(conv_k[c]/2)),
                                               nn.BatchNorm2d(conv_ch[c]), nn.ReLU()))
        self.activation = nn.Sigmoid()
        self.learning_rate = learning_rate
        self.loss = nn.L1Loss()
        self.tracker_flag = tracker
        self.tracker = None
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(mrstft_fft,
                                                            mrstft_hop,
                                                            mrstft_fft,
                                                            device=device)    # TODO: Manage device properly
        # for stft_loss in self.mrstft.stft_losses:
        #    stft_loss = stft_loss.cuda()
        self.num_bands = num_bands
        self.rate = rate
        self.loss_weights = loss_weights
        if mel_spectro:
            self.spectro = torchaudio.transforms.MelSpectrogram(n_fft=fft_size, hop_length=hop_size, sample_rate=self.rate,
                                                                power=spectro_power, n_mels=mel_num_bands)
            _, c_out, h_out, w_out = self._shape_after_conv(torch.empty(1, 1, mel_num_bands, file_size // hop_size))
        else:
            self.spectro = torchaudio.transforms.Spectrogram(n_fft=fft_size, hop_length=hop_size, power=spectro_power)
            _, c_out, h_out, w_out = self._shape_after_conv(torch.empty(1, 1, fft_size // 2 + 1, file_size // hop_size))
        self.fcl = nn.Linear(c_out * h_out * w_out, self.num_params)
        self.inv_spectro = torchaudio.transforms.InverseSpectrogram(n_fft=fft_size, hop_length=hop_size)
        self.audiologs = audiologs

    def forward(self, x, *args, **kwargs) -> Any:
        x = self.spectro(x)
        for conv in self.conv:
            x = conv(x)
        batch_size, channels, h_out, w_out = x.shape
        x = torch.flatten(x, start_dim=1)
        x = self.fcl(x)
        x = self.activation(x)
        return x

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        clean, processed, label = batch
        batch_size = processed.shape[0]
        pred = self.forward(processed)
        rec = torch.zeros(batch_size, clean.shape[-1] - 1, device=self.device)        # TODO: Remove hardcoded values
        for (i, snd) in enumerate(clean):
            for b in range(self.num_bands):
                # TODO: Make it fx agnostic
                self.mbfx.mbfx[b][0].drive_db = pred[i][b] * 50 + 10                        # TODO: how to remove hardcoded values?
                self.mbfx.mbfx[b][1].gain_db = (pred[i][self.num_bands + b] - 0.5) * 20
            rec[i] = self.mbfx(snd.detach().to(torch.device('cpu')), self.rate)
        spectral_loss = self.mrstft(rec, processed)
        self.logger.experiment.add_scalar("Spectral_loss/Train",
                                          spectral_loss, global_step=self.global_step)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
        if self.trainer.current_epoch < 25:
            self.loss_weights = [1, 0]
        elif self.trainer.current_epoch < 50:
            weight = (self.trainer.current_epoch - 25)/25
            self.loss_weights = [1 - weight, weight]
        elif self.trainer.current_epoch >= 50:
            self.loss_weights = [0, 1]
        total_loss = 10*loss*self.loss_weights[0] + spectral_loss*self.loss_weights[1]
        self.logger.experiment.add_scalar("Total_loss/Train", total_loss, global_step=self.global_step)
        return total_loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        clean, processed, label = batch
        batch_size = processed.shape[0]
        pred = self.forward(processed)
        rec = torch.zeros(batch_size, clean.shape[-1] - 1, device=self.device)  # TODO: fix hardcoded value
        for (i, snd) in enumerate(clean):
            for b in range(self.num_bands):
                # TODO: Make it fx agnostic
                self.mbfx.mbfx[b][0].drive_db = pred[i][b] * 50 + 10                    # TODO: How to remove hardcoded values?
                self.mbfx.mbfx[b][1].gain_db = (pred[i][self.num_bands + b] - 0.5) * 20
            rec[i] = self.mbfx(snd.detach().to(torch.device('cpu')), self.rate)
        spectral_loss = self.mrstft(rec, processed)
        self.logger.experiment.add_scalar("Spectral_loss/test",
                                          spectral_loss, global_step=self.global_step)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("Param_loss/test", loss, global_step=self.global_step)
        for l in range(self.audiologs):
            self.logger.experiment.add_audio(f"Audio/{l}/Original", processed[l],
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Original_params", str(label[l]), global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Matched", rec[l],
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Matched_params", str(pred[l]), global_step=self.global_step)

        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/test", scalars, global_step=self.global_step)
        total_loss = 10*loss * self.loss_weights[0] + spectral_loss * self.loss_weights[1]
        self.logger.experiment.add_scalar("Total_loss/test", total_loss, global_step=self.global_step)
        return total_loss

    def on_train_epoch_start(self) -> None:
        if self.tracker_flag and self.tracker is None:
            self.tracker = CarbonTracker(epochs=400, epochs_before_pred=10, monitor_epochs=10,
                                         log_dir=self.logger.log_dir, verbose=2)            # TODO: Remove hardcoded values
        if self.tracker_flag:
            self.tracker.epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.tracker_flag:
            self.tracker.epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)           # TODO: Remove hardcoded values
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        lr_schedulers = {"scheduler": scheduler, "interval": "epoch"}
        return {"optimizer": optimizer,"lr_scheduler": lr_schedulers}
