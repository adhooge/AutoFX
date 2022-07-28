import pathlib
from typing import Any, Optional

import pedalboard as pdb
import pytorch_lightning as pl
import sklearn.preprocessing
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

from source.classifiers.classifier_pytorch import TorchStandardScaler
from source.data.datamodules import FeaturesDataModule
from source.models.custom_distortion import CustomDistortion
from source.models.mbfx_layer import MBFxLayer
from source.multiband_fx import MultiBandFX


class SimpleMLP(pl.LightningModule):
    """A simple MLP trained directly on features."""
    def __init__(self, num_features, hidden_size, num_hidden_layers,
                 scaler_mean, scaler_std,
                 param_range_modulation, param_range_delay, param_range_disto,
                 learning_rate: float = 0.001,
                 monitor_spectral_loss: bool = False, audiologs: int = 8):
        super(SimpleMLP, self).__init__()
        num_bands = 1
        total_num_bands = 1
        self.num_features = num_features
        self.num_hidden_layers = num_hidden_layers
        if self.num_hidden_layers > 1:
            self.hidden_layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for i in range(num_hidden_layers - 1)])
        self.hidden_size = hidden_size
        modulation = MultiBandFX([pdb.Chorus], total_num_bands, device=torch.device('cpu'))
        delay = MultiBandFX([pdb.Delay], total_num_bands, device=torch.device('cpu'))
        disto = CustomDistortion()
        self.num_params = num_bands * modulation.total_num_params_per_band \
                          + num_bands * delay.total_num_params_per_band \
                          + disto.total_num_params_per_band
        self.board = [modulation, delay, disto]
        delay_layer = MBFxLayer(self.board[1], self.rate, self.param_range_delay, fake_num_bands=self.num_bands)
        modulation_layer = MBFxLayer(self.board[0], self.rate, self.param_range_modulation,
                                     fake_num_bands=self.num_bands)
        disto_layer = MBFxLayer(self.board[2], self.rate, self.param_range_disto, fake_num_bands=self.num_bands)
        self.board_layers = [modulation_layer, delay_layer, disto_layer]
        self.scaler = TorchStandardScaler()
        self.scaler.mean = torch.tensor(scaler_mean, device=torch.device('cuda'))
        self.scaler.std = torch.tensor(scaler_std, device=torch.device('cuda'))
        self.fcl_start = nn.Linear(num_features, hidden_size)
        self.fcl_end = nn.Linear(hidden_size, self.num_params)
        self.batchnorm = nn.BatchNorm1d(hidden_size, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.loss = nn.MSELoss()
        self.param_range_modulation = param_range_modulation
        self.param_range_delay = param_range_delay
        self.param_range_disto = param_range_disto
        self.activation = nn.Sigmoid()
        self.monitor_spectral_loss = monitor_spectral_loss
        self.audiologs = audiologs
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, feat, conditioning, *args, **kwargs) -> Any:
        x = torch.cat([feat, conditioning], dim=-1)
        out = self.fcl1(x)
        if self.num_hidden_layers > 1:
            out = self.hidden_layers(out)
        else:
            out = self.relu(out)
        out = self.fcl2(out)
        out = self.activation(out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        clean, processed, feat, label, conditioning, fx_class = batch
        batch_size = processed.shape[0]
        pred = self.forward(feat, conditioning)
        loss = self.loss(pred, label)
        pred_clone = pred.clone()
        # mask predictions according to fx_class
        pred_clone[:, :5] *= (fx_class[:, None] == 0)
        label[:, :5] *= (fx_class[:, None] == 0)
        pred_clone[:, 5:8] *= (fx_class[:, None] == 1)
        label[:, 5:8] *= (fx_class[:, None] == 1)
        pred_clone[:, 8:] *= (fx_class[:, None] == 2)
        label[:, 8:] *= (fx_class[:, None] == 2)
        loss = self.loss(pred_clone, label)
        pred_per_fx = [pred[:, :5], pred[:, 5:8], pred[:, 8:]]
        # loss = 0
        self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred_clone - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
        if self.monitor_spectral_loss:
            pred_per_fx = [ppf.to("cpu") for ppf in pred_per_fx]
            # self.pred = pred
            # self.pred.retain_grad()
            rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)
            for (i, snd) in enumerate(clean):
                snd_norm = snd / torch.max(torch.abs(snd))
                snd_norm = snd_norm + torch.randn_like(snd_norm) * 1e-9
                tmp = self.board_layers[fx_class[i]].forward(snd_norm.cpu(), pred_per_fx[fx_class[i]][i])
                rec[i] = tmp.clone() / torch.max(torch.abs(tmp))
            target_normalized = processed[:, 0, :] / torch.max(torch.abs(processed[:, 0, :]), dim=-1, keepdim=True)[0]
            pred_normalized = rec.to(self.device)
            spec_loss = self.spectral_loss(pred_normalized, target_normalized)
            features = self.compute_features(pred_normalized)
            feat_loss = self.feat_loss(features, feat)
            spectral_loss = self.feat_weight * feat_loss + self.mrstft_weight * spec_loss
        else:
            spectral_loss = 0
            spec_loss = 0
            feat_loss = 0
        self.logger.experiment.add_scalar("Feature_loss/Train",
                                          feat_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("MRSTFT_loss/Train",
                                          spec_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("Total_Spectral_loss/Train",
                                          spectral_loss, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        clean, processed, feat, label, conditioning, fx_class = batch
        batch_size = processed.shape[0]
        pred = self.forward(feat, conditioning)
        loss = self.loss(pred, label)
        pred_clone = pred.clone()
        # mask predictions according to fx_class
        pred_clone[:, :5] *= (fx_class[:, None] == 0)
        label[:, :5] *= (fx_class[:, None] == 0)
        pred_clone[:, 5:8] *= (fx_class[:, None] == 1)
        label[:, 5:8] *= (fx_class[:, None] == 1)
        pred_clone[:, 8:] *= (fx_class[:, None] == 2)
        label[:, 8:] *= (fx_class[:, None] == 2)
        loss = self.loss(pred_clone, label)
        pred_per_fx = [pred[:, :5], pred[:, 5:8], pred[:, 8:]]
        # loss = 0
        self.logger.experiment.add_scalar("Param_loss/Train", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred_clone - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
        if self.monitor_spectral_loss:
            pred_per_fx = [ppf.to("cpu") for ppf in pred_per_fx]
            # self.pred = pred
            # self.pred.retain_grad()
            rec = torch.zeros(batch_size, clean.shape[-1], device=self.device)
            for (i, snd) in enumerate(clean):
                snd_norm = snd / torch.max(torch.abs(snd))
                snd_norm = snd_norm + torch.randn_like(snd_norm) * 1e-9
                tmp = self.board_layers[fx_class[i]].forward(snd_norm.cpu(), pred_per_fx[fx_class[i]][i])
                rec[i] = tmp.clone() / torch.max(torch.abs(tmp))
            target_normalized = processed[:, 0, :] / torch.max(torch.abs(processed[:, 0, :]), dim=-1, keepdim=True)[0]
            pred_normalized = rec.to(self.device)
            spec_loss = self.spectral_loss(pred_normalized, target_normalized)
            features = self.compute_features(pred_normalized)
            feat_loss = self.feat_loss(features, feat)
            spectral_loss = self.feat_weight * feat_loss + self.mrstft_weight * spec_loss
        else:
            spectral_loss = 0
            spec_loss = 0
            feat_loss = 0
        self.logger.experiment.add_scalar("Feature_loss/Train",
                                          feat_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("MRSTFT_loss/Train",
                                          spec_loss, global_step=self.global_step)
        self.logger.experiment.add_scalar("Total_Spectral_loss/Train",
                                          spectral_loss, global_step=self.global_step)
        return loss

    def on_validation_end(self) -> None:
        clean, processed, feat, label, conditioning, fx_class = next(iter(self.trainer.val_dataloaders[0]))
        conditioning = conditioning.to(self.device)
        fx_class = fx_class.to(self.device)
        pred = self.forward(feat.to(self.device), conditioning=conditioning)
        pred = pred.to("cpu")
        pred_per_fx = [pred[:, :5], pred[:, 5:8], pred[:, 8:]]
        rec = torch.zeros(clean.shape[0], clean.shape[-1], device=self.device)  # TODO: fix hardcoded value
        # features = self.compute_features(processed[:, 0, :].to(self.device))
        for (i, snd) in enumerate(clean):
            rec[i] = self.board_layers[fx_class[i]].forward(snd, pred_per_fx[fx_class[i]][i])
        for l in range(self.audiologs):
            self.logger.experiment.add_text(f"Audio/{l}/Original_feat",
                                            str(feat[l]), global_step=self.global_step)
            # self.logger.experiment.add_text(f"Audio/{l}/Predicted_feat",
            #                                 str(features[l]), global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Original", processed[l] / torch.max(torch.abs(processed[l])),
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_audio(f"Audio/{l}/Matched", rec[l] / torch.max(torch.abs(rec[l])),
                                             sample_rate=self.rate, global_step=self.global_step)
            self.logger.experiment.add_text(f"Audio/{l}/Predicted_params", str(pred[l]), global_step=self.global_step)
            if not self.out_of_domain:
                self.logger.experiment.add_text(f"Audio/{l}/Matched_params", str(pred[l]), global_step=self.global_step)
                self.logger.experiment.add_text(f"Audio/{l}/Original_params", str(label[l]),
                                                global_step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    df = pd.read_csv("/home/alexandre/dataset/modulation_delay_distortion_guitar_mono_cut/data.csv", index_col=0)
    CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
    PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/modulation_delay_distortion_guitar_mono_cut")
    OUT_OF_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050_cut")
    NUM_FEATURES = 148
    HIDDEN_SIZE = 100
    NUM_HIDDEN_LAYERS = 10
    PARAM_RANGE_DISTORTION = [(0, 60),
                              (50, 500), (-10, 10), (0.5, 2),
                              (500, 2000), (-10, 10), (0.5, 2)]
    PARAM_RANGE_DELAY = [(0, 1), (0, 1), (0, 1)]
    PARAM_RANGE_MODULATION = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
    data = pd.read_csv(PROCESSED_PATH / 'data_mlp.csv')
    scaler = sklearn.preprocessing.StandardScaler()
    FEAT_COL = None
    scaler.fit(data[[FEAT_COL]])

    datamodule = FeaturesDataModule(CLEAN_PATH, PROCESSED_PATH, OUT_OF_DOMAIN_PATH,
                                    in_scaler_mean=scaler.mean, in_scaler_std=scaler.std,
                                    out_scaler_mean=scaler.mean, out_scaler_std=scaler.std,
                                    seed=2, batch_size=64,
                                    conditioning=True, classes2keep=[0, 1, 2])

    mlp = SimpleMLP(NUM_FEATURES, HIDDEN_SIZE, NUM_HIDDEN_LAYERS,
                    scaler.mean, scaler.std, PARAM_RANGE_MODULATION,
                    PARAM_RANGE_DELAY, PARAM_RANGE_DISTORTION,
                    monitor_spectral_loss=True)
    logger = TensorBoardLogger("/home/alexandre/logs/SimpleMLP", name="29July")
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=1000, log_every_n_steps=100)

    trainer.fit(mlp, datamodule=datamodule)