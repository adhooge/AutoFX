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


def _settings_list2dict(settings_list, fx: pdb.Plugin):
    settings_dict = {}
    items = list(fx.__dict__.items())
    cnt = 0
    for item in items:
        if isinstance(item[1], property):
            settings_dict[item[0]] = settings_list[cnt]
            cnt += 1
    return settings_dict


class MultiBandFX:
    def __call__(self, audio, rate, *args, **kwargs):
        return self.process(audio, rate, args, kwargs)

    def __init__(self, fx: pdb.Plugin or list[pdb.Plugin], bands: int | list[float] | list[Tuple],
                 device: torch.device = torch.device('cpu'), attenuation: int = 100):
        """
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
                if isinstance(fx[0], pdb.Plugin):
                    board = []
                    for f in fx:
                        params = util.get_fx_params(f)
                        tmp = f.__class__()
                        tmp = util.set_fx_params(tmp, params)
                        board.append(tmp[0])
                else:
                    board = [plug() for plug in fx]
                board = pdb.Pedalboard(board)
                self.mbfx.append(board)
            else:
                tmp = fx()
                if params is not None:
                    tmp = util.set_fx_params(tmp, params)
                self.mbfx.append(tmp)
        self.device = device
        if int(np.log2(self.num_bands)) == np.log2(self.num_bands):
            self.filter_bank = PQMF(attenuation, self.num_bands, polyphase=True,
                                    device=device)  # TODO: Fix hardcoded device
        else:
            self.filter_bank = PQMF(attenuation, self.num_bands, polyphase=False, device=device)

    def set_fx_params(self, params: list[dict] or dict or list) -> None:
        # TODO: Manage all possible cases. As of now, only complete setting of parameters is allowed
        params = np.array(params)
        if params.ndim < 2 or (params.ndim == 2 and not isinstance(params[0, 0], dict)):
            raise NotImplementedError
        else:
            for b in range(self.num_bands):
                if self.num_fx >= 2 and params.ndim == 3:
                    board_settings = [_settings_list2dict(params[b][f], self.fx_per_band[f])
                                      for f in range(self.num_fx)]
                else:
                    board_settings = params[b]
                self.mbfx[b] = util.set_fx_params(self.mbfx[b], board_settings)

    def add_perturbation_to_fx_params(self, perturbation):
        # TODO: Ensure parameters do not exceed limits
        # TODO: Deal with conversion between 0/1 and min/max
        raise NotImplementedError

    @property
    def fx_per_band(self):
        return [self.mbfx[0][i].__class__ for i in range(self.num_fx)]

    @property
    def num_fx(self):
        return len(self.mbfx[0])

    @property
    def settings(self):
        settings = []
        for fx in self.mbfx:
            settings.append(util.get_fx_params(fx))
        return settings

    @property
    def settings_list(self):
        settings_dict = self.settings
        settings_list = []
        for band in settings_dict:
            tmp = []
            for dico in band:
                tmp.append(list(dico.values()))
            settings_list.append(tmp)
        return settings_list

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
