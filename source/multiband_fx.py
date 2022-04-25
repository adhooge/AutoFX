"""
Class to define an FX on several frequency bands with their own parameters.
Extends the pedalboard.Plugin class
"""
from typing import Tuple

import pedalboard as pdb
import numpy as np
import torch
from torch import double
import util

from numba import jit, cuda

from rave_pqmf import PQMF


class MultiBandFX:
    def __call__(self, audio, rate, *args, **kwargs):
        return self.process(audio, rate, args, kwargs)

    def __init__(self, fx: pdb.Plugin or list[pdb.Plugin], bands: int | list[float] | list[Tuple],
                 device: torch.device = torch.device('cpu'), attenuation: int = 100):
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
            params = util.get_fx_params(fx)
            fx = fx.__class__
        else:
            params = None
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
            if isinstance(fx, list):
                board = pdb.Pedalboard([plug() for plug in fx])
                self.mbfx.append(board)
            else:
                tmp = fx()
                if params is not None:
                    tmp = util.set_fx_params(tmp, params)
                self.mbfx.append(tmp)
        self.device = device
        if int(np.log2(self.num_bands)) == np.log2(self.num_bands):
            self.filter_bank = PQMF(attenuation, self.num_bands, polyphase=True, device=device)     # TODO: Fix hardcoded device
        else:
            self.filter_bank = PQMF(attenuation, self.num_bands, polyphase=False, device=device)

    def _settings_list2dict(self):
        # TODO
        return NotImplemented

    def _settings_dict2list(self):
        # TODO
        return NotImplemented

    def set_fx_params(self, params: list[dict] or dict) -> None:
        raise NotImplementedError       # TODO

    def add_perturbation_to_fx_params(self, perturbation):
        # TODO: Ensure parameters do not exceed limits
        # TODO: Deal with conversion between 0/1 and min/max
        raise NotImplementedError

    @property
    def settings(self):
        settings = []
        for (b, fx) in enumerate(self.mbfx):
            settings.append(util.get_fx_params(fx))
        return settings

    @property
    def settings_list(self):
        #TODO
        return NotImplemented

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
