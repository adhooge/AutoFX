import pathlib
import warnings

from multiband_fx import MultiBandFX
import util
import random
import pedalboard as pdb
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

# warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = pathlib.Path("/home/alexandre/dataset/full_idmt_dry")
NUM_BANDS = 4
FX = [pdb.Distortion, pdb.Gain]
DRIVE_MIN = 10
DRIVE_MAX = 60
GAIN_MIN = -10
GAIN_MAX = 10
OUT_PATH = pathlib.Path("/home/alexandre/dataset/mbfx_disto")

NUM_RUNS = 4

dataframe = pd.DataFrame(columns=['drive_level0', 'drive_level1', 'drive_level2', 'drive_level3',
                                  'gain_level0', 'gain_level1', 'gain_level2', 'gain_level3'], dtype='float64')
mbfx = MultiBandFX(FX, NUM_BANDS)
for i in tqdm(range(NUM_RUNS), position=1):
    for file in tqdm(DATA_PATH.iterdir(), total=5004, position=0, leave=True):
        drive_levels = [random.uniform(DRIVE_MIN, DRIVE_MAX) for b in range(NUM_BANDS)]
        gain_levels = [random.uniform(GAIN_MIN, GAIN_MAX) for b in range(NUM_BANDS)]
        drive_levels.extend(gain_levels)
        for b in range(NUM_BANDS):
            mbfx.mbfx[b][0].drive_db = drive_levels[b]
            mbfx.mbfx[b][1].gain_db = gain_levels[b]
        audio, rate = util.read_audio(file, normalize=True, cut_beginning=0.45, add_noise=True)
        audio = mbfx.process(audio, rate)
        dataframe.loc[file.stem + '_' + str(i)] = drive_levels
        sf.write(OUT_PATH / (file.stem + '_' + str(i) + file.suffix), audio.T, int(rate))
    dataframe.to_csv(OUT_PATH / "params.csv")
    dataframe.to_pickle(OUT_PATH / "params.pkl")



