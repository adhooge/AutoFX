"""
Class to define an FX on several frequency bands with their own parameters.
Extends the pedalboard.Plugin class
"""
from typing import Tuple

import pedalboard as pdb
import numpy as np
import torch
from torch import double

from rave_pqmf import PQMF


class MultiBandFX:
    def __call__(self, audio, rate, *args, **kwargs):
        return self.process(audio, rate, args, kwargs)

    def __init__(self, fx: pdb.Plugin, bands: int | list[float] | list[Tuple], attenuation: int = 100):
        """
        TODO: Currently, parameters of FX are not transferred to MBFX so MBFX is always initialized to default
        :param fx: An effect from Pedalboard to use in a multiband fashion
        :param bands: If integer: number of frequency bands to use.
                      If list: normalized frequencies separating bands. Number of bands is len(list) + 1.
                      Example: [0.25, 0.4] will create 3 frequency bands:
                      One from 0 to 0.25, one from 0.25 to 0.4 and one from 0.4 to 0.5.
                      Bands can also be directly specified as [(0, 0.25), (0.25, 0.5)]. They should always start at 0
                      and finish at 0.5.
        :param attenuation: attenuation of the filter bank. Should be an int from between 80 and 120.
        """
        if isinstance(fx, pdb.Plugin):
            fx = fx.__class__
        if isinstance(bands, int):
            self.num_bands = bands
            self.bands = []
            freq = np.linspace(0, 0.5, self.num_bands + 1)
            for f in range(self.num_bands):
                self.bands.append((freq[f], freq[f + 1]))
        elif isinstance(bands, list):
            if isinstance(bands[0], float):
                self.num_bands = len(bands) + 1
                self.bands = []
                self.bands.append((0, bands[0]))
                for f in range(1, len(bands) - 1):
                    self.bands.append((bands[f], bands[f + 1]))
                self.bands.append((bands[-1], 0.5))
            elif isinstance(bands[0], Tuple):
                self.num_bands = len(bands)
                if bands[0][0] != 0:
                    raise ValueError("First band should start at 0.")
                if bands[-1][-1] != 0.5:
                    raise ValueError("Last band should end at 0.5")
                self.bands = bands
            else:
                raise TypeError("Format not recognized for bands.")
        else:
            raise TypeError("Cannot create frequency bands. Check argument.")
        self.mbfx = []
        for i in range(self.num_bands):
            self.mbfx.append(fx())
        if int(np.log2(self.num_bands)) == np.log2(self.num_bands):
            self.filter_bank = PQMF(attenuation, self.num_bands, polyphase=True)
        else:
            self.filter_bank = PQMF(attenuation, self.num_bands, polyphase=False)

    @property
    def settings(self):
        settings = []
        for (b, fx) in enumerate(self.mbfx):
            fx_settings = {}
            items = list(fx.__class__.__dict__.items())
            for item in items:
                if isinstance(item[1], property):
                    fx_settings[item[0]] = item[1].__get__(fx, fx.__class__)
            settings.append(fx_settings)
        return settings

    def process(self, audio, rate, *args, **kwargs):
        """
        TODO: Make it cleaner between Torch and numpy. Managing Batch sizes properly
        :param audio:
        :param rate:
        :param args:
        :param kwargs:
        :return:
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if audio.dim() == 2:
            audio = audio[None, :]
        audio_bands = self.filter_bank.forward(audio)[0]
        for (b, fx) in enumerate(self.mbfx):
            audio_bands[b] = torch.from_numpy(fx(audio_bands[b], rate))
        out = self.filter_bank.inverse(audio_bands[None, :, :])
        return out[0]
