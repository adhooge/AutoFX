import pathlib
import warnings

import numpy as np

from multiband_fx import MultiBandFX
import util
import random
import pedalboard as pdb
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

# warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry")
NUM_BANDS = 32
NUM_CHANGED_BANDS = 4
FX = [pdb.Distortion(0), pdb.Gain(0)]
DRIVE_MIN = 10
DRIVE_MAX = 60
GAIN_MIN = -10
GAIN_MAX = 10
OUT_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto_guitar_mono_int_32")

NUM_RUNS = 20

# TODO: Could automatically adapt to new FX

dataframe = pd.DataFrame(columns=['drive_level0', 'drive_level1', 'drive_level2', 'drive_level3',
                                  'gain_level0', 'gain_level1', 'gain_level2', 'gain_level3'], dtype='float64')
mbfx = MultiBandFX(FX, NUM_BANDS)
if (OUT_PATH / "params.csv").exists():
    raise ValueError("Output directory already has a params.csv file. Aborting.")
for i in tqdm(range(NUM_RUNS), position=1):
    for file in tqdm(DATA_PATH.iterdir(), total=1872, position=0, leave=True):
        drive_levels = np.array([random.randint(DRIVE_MIN, DRIVE_MAX) for b in range(NUM_CHANGED_BANDS)])
        gain_levels = np.array([random.randint(GAIN_MIN, GAIN_MAX) for b in range(NUM_CHANGED_BANDS)])
        normalized_levels = np.hstack(((drive_levels - 10)/50, (gain_levels/20 + 0.5)))
        for b in range(NUM_CHANGED_BANDS):
            mbfx.mbfx[b][0].drive_db = drive_levels[b]
            mbfx.mbfx[b][1].gain_db = gain_levels[b]
        audio, rate = util.read_audio(file, normalize=True, add_noise=True)
        audio = mbfx.process(audio, rate)
        dataframe.loc[file.stem + '_' + str(i)] = normalized_levels
        audio = audio[0, ::2]
        sf.write(OUT_PATH / (file.stem + '_' + str(i) + file.suffix), audio, int(rate//2))
    dataframe.to_csv(OUT_PATH / "params.csv")
    dataframe.to_pickle(OUT_PATH / "params.pkl")



