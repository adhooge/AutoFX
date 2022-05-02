"""
Dataset class for sounds whose audio features have been extracted
"""
import pathlib
from typing import List

import pedalboard as pdb

import torch
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from multiband_fx import MultiBandFX
import util


class FeaturesDataset(Dataset):
    def __init__(self, params_file: str or pathlib.Path, features_file: str or pathlib.Path,
                 cln_snd_dir: str or pathlib.Path, prc_snd_dir: str or pathlib.Path, rate: int,
                 resampling_rate: int = 22050, **kwargs):
        self.snd_labels = pd.read_csv(params_file)
        self.cln_snd_dir = pathlib.Path(cln_snd_dir)
        self.prc_snd_dir = pathlib.Path(prc_snd_dir)
        self.features = pd.read_csv(features_file)
        self.transform = nn.Sequential(
            torchaudio.transforms.Resample(rate, resampling_rate)
        )

    def __len__(self):
        return len(self.snd_labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        cln_snd_path = self.cln_snd_dir / (self.snd_labels.iloc[item, 0].split('_')[0] + '.wav')
        prc_snd_path = self.prc_snd_dir / (self.snd_labels.iloc[item, 0] + '.wav')
        cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
        prc_sound, rate = torchaudio.load(prc_snd_path, normalize=True)
        params = self.snd_labels.iloc[item, 1:]
        params = torch.Tensor(params)
        features = self.features.iloc[item, 1:-1]
        features = torch.Tensor(features)
        cln_resampled = self.transform(cln_sound)
        prc_resampled = self.transform(prc_sound)
        return cln_resampled, prc_resampled, params, features
