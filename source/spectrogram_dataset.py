"""
Dataset class for spectrogram from audio.
"""
import pathlib
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset


class SpectroDataset(Dataset):
    def __init__(self, labels_file: str or pathlib.Path,
                 snd_dir: str or pathlib.Path, idmt: bool = False, **kwargs):
        self.snd_labels = pd.read_csv(labels_file)
        self.snd_dir = pathlib.Path(snd_dir)
        self.idmt = idmt
        self.transform = torchaudio.transforms.Spectrogram(**kwargs)

    def __len__(self):
        return len(self.snd_labels)

    def __getitem__(self, item):
        snd_path = self.snd_dir / (str(self.snd_labels.iloc[item, 0]) + '.wav')
        sound, rate = torchaudio.load(snd_path, normalize=True)
        label = self.snd_labels.iloc[item, 1]
        if self.idmt:
            sound = sound[:, int(0.45*rate):]
        sound = torchaudio.transforms.Resample(rate, 16000)
        spectro = self.transform(sound)
        return spectro, label
