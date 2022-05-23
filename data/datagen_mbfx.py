import os
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
FX = [pdb.Reverb()]
DRIVE_MIN = 10
DRIVE_MAX = 60
GAIN_MIN = -10
GAIN_MAX = 10
RATE_MIN = 0.1
RATE_MAX = 10
DEPTH_MIN = 0
DEPTH_MAX = 1
CENTRE_DELAY_MIN = 0
CENTRE_DELAY_MAX = 20
FEEDBACK_MIN = 0
FEEDBACK_MAX = 1
MIX_MIN = 0
MIX_MAX = 1
PARAM_RANGE = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0)]
PARAMS = ["p-room_size", "p-damping", "p-wet_level", "p-dry_level", "p-width"]
OUT_PATH = pathlib.Path("/home/alexandre/dataset/reverb_gen_cut")

NUM_RUNS = 20

# TODO: Could automatically adapt to new FX

dataframe = pd.DataFrame(columns=PARAMS, dtype='float64')
mbfx = MultiBandFX(FX, NUM_BANDS)
if not(OUT_PATH.exists()):
    os.mkdir(OUT_PATH)
if (OUT_PATH / "params.csv").exists():
    raise ValueError("Output directory already has a params.csv file. Aborting.")
for i in tqdm(range(NUM_RUNS), position=1):
    for file in tqdm(DATA_PATH.iterdir(), total=1872, position=0, leave=True):
        # drive_levels = np.array([random.randint(DRIVE_MIN, DRIVE_MAX) for b in range(NUM_CHANGED_BANDS)])
        # gain_levels = np.array([random.randint(GAIN_MIN, GAIN_MAX) for b in range(NUM_CHANGED_BANDS)])
        # rate = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        # depth = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        # centre_delay = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        # feedback = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        # mix = np.array([round(random.random(), 2) for b in range(NUM_CHANGED_BANDS)])
        # normalized_levels = np.hstack(((drive_levels - 10)/50, (gain_levels/20 + 0.5)))
        # params = np.hstack((rate, depth, centre_delay, feedback, mix))
        params = [round(random.random(), 2) for p in range(len(PARAMS))]
        params.append(0)    #  freeze_mode for reverb
        mbfx.set_fx_params(params, param_range=PARAM_RANGE, flat=True)
        # for b in range(NUM_CHANGED_BANDS):
            # mbfx.mbfx[b][0].drive_db = drive_levels[b]
            # mbfx.mbfx[b][1].gain_db = gain_levels[b]
        #    mbfx.mbfx[b][0].rate_hz = rate[b] * (RATE_MAX - RATE_MIN) + RATE_MIN
        #    mbfx.mbfx[b][0].depth = depth[b] * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
        #    mbfx.mbfx[b][0].centre_delay_ms = centre_delay[b] * (CENTRE_DELAY_MAX - CENTRE_DELAY_MIN) + CENTRE_DELAY_MIN
        #    mbfx.mbfx[b][0].feedback = feedback[b] * (FEEDBACK_MAX - FEEDBACK_MIN) + FEEDBACK_MIN
        #    mbfx.mbfx[b][0].mix = mix[b] * (MIX_MAX - MIX_MIN) + MIX_MIN
        audio, rate = util.read_audio(file, normalize=True, add_noise=True)
        processed = np.zeros((1, int(5.12*rate)))   # zero-pad to keep reverb tail (up to 5s)
        processed[0, :len(audio[0])] = audio
        processed = mbfx.process(processed, rate)
        dataframe.loc[file.stem + '_' + str(i)] = params[:-1]
        # audio = audio[0, ::2]
        processed = processed[0]
        sf.write(OUT_PATH / (file.stem + '_' + str(i) + file.suffix), processed, int(rate//1))      # CAREFUL WITH SUBSAMPLING
    dataframe.to_csv(OUT_PATH / "params.csv")
    dataframe.to_pickle(OUT_PATH / "params.pkl")



