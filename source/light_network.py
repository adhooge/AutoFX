from typing import Any, Optional

import pedalboard
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LightNetwork(pl.LightningModule):
    def _change_fx_settings(self, settings, param_range):
        # TODO
        pass

    def __init__(self, input_features: int, hidden_layer_size: int,
                 output_size: int, fx: pedalboard.Plugin, param_range):
        super(LightNetwork, self).__init__()
        self.in_feat = input_features
        self.hidden_size = hidden_layer_size
        self.out_size = output_size
        self.linear1 = nn.Linear(self.in_feat, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.out_size)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(self.in_feat)
        self.loss = nn.MSELoss()
        self.fx = fx
        self.param_range = param_range

    def forward(self, x, *args, **kwargs) -> Any:
        out = self.norm(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.activation(out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        feat, label = batch
        pred = self.forward(feat)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("Loss/Train", loss, global_step=self.global_step)
        return loss

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        # TODO: Copy from AutoFXRESNET
        # TODO: Add second dataloader for testing in and out-of-domain while training
        pass
