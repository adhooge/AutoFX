"""
Class to define an FX on several frequency bands with their own parameters.
Extends the pedalboard.Plugin class
"""
from typing import Tuple

import pedalboard as pdb
import numpy as np


class MultiBandFX(pdb.Plugin):
    def __init__(self, fx: pdb.Plugin, bands: int | list[float] | list[Tuple]):
        """
        :param fx: An effect from Pedalboard to use in a multiband fashion
        :param bands: If integer: number of frequency bands to use.
                      If list: normalized frequencies separating bands. Number of bands is len(list) + 1.
                      Example: [0.25, 0.4] will create 3 frequency bands:
                      One from 0 to 0.25, one from 0.25 to 0.4 and one from 0.4 to 0.5.
                      Bands can also be directly specified as [(0, 0.25), (0.25, 0.5)]. They should always start at 0
                      and finish at 0.5.
        """
        super().__init__()
        if isinstance(bands, int):
            self.num_bands = bands
            self.bands = []
            freq = np.linspace(0, 0.5, self.num_bands + 1)
            for f in range(self.num_bands):
                self.bands.append((freq[2*f], freq[2*f + 1]))
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
        self.mbfx = [fx for i in range(self.num_bands)]
        self.settings = [{}] * self.num_bands
        self._init_settings(fx)

    def _init_settings(self, fx: pdb.Plugin):
        items = list(fx.__class__.__dict__.items())
        settings = {}
        for item in items:
            if isinstance(item[1], property):
                settings[item[0]] = item[1].__get__(fx, fx.__classs__)
        for b in range(self.num_bands):
            self.settings[b] = settings

    def process(self, *args, **kwargs):
        """
        TODO
        :param args:
        :param kwargs:
        :return:
        """
        return NotImplemented

