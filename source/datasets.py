import torch
import pathlib
import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class FeatureInDomainDataset(Dataset):
    def __init__(self, data_path: pathlib.Path or str, validation: bool = False,
                 clean_path: pathlib.Path or str = None, processed_path: pathlib.Path or str = None):
        if validation and (clean_path is None or processed_path is None):
            raise ValueError("Clean and Processed required for validation dataset.")
        self.data_path = pathlib.Path(data_path)
        self.validation = validation
        self.clean_path = pathlib.Path(clean_path) if clean_path is not None else clean_path
        self.processed_path = pathlib.Path(processed_path) if processed_path is not None else processed_path
        self.data = pd.read_csv(data_path / "data.csv")
        columns = list(self.data.columns)
        num_features = 0
        num_param = 0
        for c in columns:
            if 'f-' in c:
                num_features += 1
            elif 'p-' in c:
                num_param += 1
        self.num_features = num_features
        self.num_param = num_param

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filename = self.data.iloc[item, 0]
        if self.validation:
            cln_snd_path = self.clean_path / (filename.split('_')[0] + '.wav')
            prc_snd_path = self.processed_path / (filename + '.wav')
            cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
            prc_sound, rate = torchaudio.load(prc_snd_path, normalize=True)
            cln_resampled = self.transform(cln_sound)
            prc_resampled = self.transform(prc_sound)
        params = self.data.iloc[item, -self.num_param:]
        params = torch.Tensor(params)
        features = self.data.iloc[item, 1:1 + self.num_features]
        features = torch.Tensor(features)
        if self.validation:
            return cln_resampled, prc_resampled, features, params
        else:
            return features, params


class FeatureOutDomainDataset(Dataset):
    def __init__(self, data_path: pathlib.Path or str,
                 clean_path: pathlib.Path or str = None, processed_path: pathlib.Path or str = None):
        self.data_path = pathlib.Path(data_path)
        self.clean_path = pathlib.Path(clean_path) if clean_path is not None else clean_path
        self.processed_path = pathlib.Path(processed_path) if processed_path is not None else processed_path
        self.data = pd.read_csv(data_path / "data.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filename = self.data.iloc[item, 0]
        cln_snd_path = self.clean_path / (filename.split('_')[0] + '.wav')
        prc_snd_path = self.processed_path / (filename + '.wav')
        cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
        prc_sound, rate = torchaudio.load(prc_snd_path, normalize=True)
        cln_resampled = self.transform(cln_sound)
        prc_resampled = self.transform(prc_sound)
        features = self.data.iloc[item, 1:]
        features = torch.Tensor(features)
        return cln_resampled, prc_resampled, features
