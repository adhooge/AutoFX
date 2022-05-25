import torch
import pathlib
import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class FeatureInDomainDataset(Dataset):
    def __init__(self, data_path: pathlib.Path or str, validation: bool = False,
                 clean_path: pathlib.Path or str = None, processed_path: pathlib.Path or str = None,
                 pad_length: int = None, reverb: bool=False):
        if validation and (clean_path is None or processed_path is None):
            raise ValueError("Clean and Processed required for validation dataset.")
        self.data_path = pathlib.Path(data_path)
        self.validation = validation
        self.clean_path = pathlib.Path(clean_path) if clean_path is not None else None
        self.processed_path = pathlib.Path(processed_path) if processed_path is not None else None
        self.data = pd.read_csv(data_path / "data.csv", index_col=0)
        columns = list(self.data.columns)
        num_features = 0
        num_param = 0
        features_columns = []
        param_columns = []
        for (i, c) in enumerate(columns):
            if 'f-' in c:
                num_features += 1
                features_columns.append(i)
            elif 'p-' in c:
                num_param += 1
                param_columns.append(i)
        self.num_features = num_features
        self.feat_columns = features_columns
        self.num_param = num_param
        self.param_columns = param_columns
        if pad_length is None:
            if reverb:
                self.pad_length = 2**17
            else:
                self.pad_length = 35000
        else:
            self.pad_length = pad_length
        self.reverb = reverb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filename = self.data.iloc[item, 0]
        if self.validation:
            cln_snd_path = self.clean_path / (filename.split('_')[0] + '.wav')
            prc_snd_path = self.processed_path / (filename + '.wav')
            cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
            prc_sound, rate = torchaudio.load(prc_snd_path, normalize=True)
            cln_sound = cln_sound[0]
            prc_sound = prc_sound[0]
            cln_pad = torch.zeros((1, self.pad_length))
            cln_pad[0, :len(cln_sound)] = cln_sound
            cln_pad[0, len(cln_sound):] = torch.randn(self.pad_length-len(cln_sound)) / 1e9
            prc_pad = torch.zeros((1, self.pad_length))
            prc_pad[0, :len(prc_sound)] = prc_sound
            prc_pad[0, len(prc_sound):] = torch.randn(self.pad_length - len(prc_sound)) / 1e9
        # print(self.data.iloc[item])
        params = self.data.iloc[item, self.param_columns]
        params = torch.Tensor(params)
        if self.reverb:
            params = torch.hstack([params, torch.zeros(1)])
        # print(params)
        features = self.data.iloc[item, self.feat_columns]
        features = torch.Tensor(features)
        # print(features)
        if self.validation:
            return cln_pad, prc_pad, features, params
        else:
            return features, params


class FeatureOutDomainDataset(Dataset):
    def __init__(self, data_path: pathlib.Path or str,
                 clean_path: pathlib.Path or str = None, processed_path: pathlib.Path or str = None,
                 pad_length: int = 35000, index_col: int = None):
        self.data_path = pathlib.Path(data_path)
        self.clean_path = pathlib.Path(clean_path) if clean_path is not None else clean_path
        self.processed_path = pathlib.Path(processed_path) if processed_path is not None else processed_path
        self.data = pd.read_csv(data_path / "data.csv", index_col=index_col)
        self.fx2clean = pd.read_csv(data_path / "fx2clean.csv")
        self.pad_length = pad_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filename = self.data.iloc[item, 0]
        cln_snd_path = self.clean_path / (self.fx2clean.iloc[item, 1] + '.wav')
        fx_snd_path = self.processed_path / (self.fx2clean.iloc[item, 0] + '.wav')
        cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
        prc_sound, rate = torchaudio.load(fx_snd_path, normalize=True)
        cln_sound = cln_sound[0]
        prc_sound = prc_sound[0]
        cln_pad = torch.zeros((1, self.pad_length))
        cln_pad[0, :len(cln_sound)] = cln_sound
        cln_pad[0, len(cln_sound):] = torch.randn(self.pad_length - len(cln_sound)) / 1e9
        prc_pad = torch.zeros((1, self.pad_length))
        prc_pad[0, :len(prc_sound)] = prc_sound
        prc_pad[0, len(prc_sound):] = torch.randn(self.pad_length - len(prc_sound)) / 1e9
        features = self.data.iloc[item, 1:]
        features = torch.Tensor(features)
        return cln_pad, prc_pad, features
