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

DATA_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
NUM_BANDS = 1
NUM_CHANGED_BANDS = 1
FX = [pdb.Chorus()]
DRIVE_MIN = 10
DRIVE_MAX = 60
GAIN_MIN = -10
GAIN_MAX = 10
RATE_MIN = 0.5
RATE_MAX = 2
DEPTH_MIN = 0
DEPTH_MAX = 1
CENTRE_DELAY_MIN = 1
CENTRE_DELAY_MAX = 10
FEEDBACK_MIN = 0
FEEDBACK_MAX = 1
MIX_MIN = 0
MIX_MAX = 1
OUT_PATH = pathlib.Path("/home/alexandre/dataset/modulation_guitar_mono_cut")

NUM_RUNS = 20

# TODO: Could automatically adapt to new FX

dataframe = pd.DataFrame(columns=["p-rate_hz", "p-depth", "p-centre_delay_ms", "p-feedback", "p-mix"], dtype='float64')
mbfx = MultiBandFX(FX, NUM_BANDS)
if (OUT_PATH / "params.csv").exists():
    raise ValueError("Output directory already has a params.csv file. Aborting.")
for i in tqdm(range(NUM_RUNS), position=1):
    for file in tqdm(DATA_PATH.iterdir(), total=1872, position=0, leave=True):
        # drive_levels = np.array([random.randint(DRIVE_MIN, DRIVE_MAX) for b in range(NUM_CHANGED_BANDS)])
        # gain_levels = np.array([random.randint(GAIN_MIN, GAIN_MAX) for b in range(NUM_CHANGED_BANDS)])
        rate = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        depth = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        centre_delay = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        feedback = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        mix = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        # normalized_levels = np.hstack(((drive_levels - 10)/50, (gain_levels/20 + 0.5)))
        params = np.hstack((rate, depth, centre_delay, feedback, mix))
        for b in range(NUM_CHANGED_BANDS):
            # mbfx.mbfx[b][0].drive_db = drive_levels[b]
            # mbfx.mbfx[b][1].gain_db = gain_levels[b]
            mbfx.mbfx[b][0].rate_hz = rate[b] * (RATE_MAX - RATE_MIN) + RATE_MIN
            mbfx.mbfx[b][0].depth = depth[b] * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
            mbfx.mbfx[b][0].centre_delay_ms = centre_delay[b] * (CENTRE_DELAY_MAX - CENTRE_DELAY_MIN) + CENTRE_DELAY_MIN
            mbfx.mbfx[b][0].feedback = feedback[b] * (FEEDBACK_MAX - FEEDBACK_MIN) + FEEDBACK_MIN
            mbfx.mbfx[b][0].mix = mix[b] * (MIX_MAX - MIX_MIN) + MIX_MIN
        audio, rate = util.read_audio(file, normalize=True, add_noise=True)
        audio = mbfx.process(audio, rate)
        dataframe.loc[file.stem + '_' + str(i)] = params
        audio = audio[0, ::2]
        sf.write(OUT_PATH / (file.stem + '_' + str(i) + file.suffix), audio, int(rate//2))
    dataframe.to_csv(OUT_PATH / "params.csv")
    dataframe.to_pickle(OUT_PATH / "params.pkl")



