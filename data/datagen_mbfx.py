import argparse
import os
import pathlib
import sys
import warnings

import numpy as np
import torch

import source.util as util
import random
import pedalboard as pdb
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm
from source.models.custom_distortion import CustomDistortion

# warnings.filterwarnings("ignore", category=FutureWarning)
def main(parser):
    args = vars(parser.parse_args())
    DATA_PATH = pathlib.Path(args['data-path'])
    OUT_PATH = pathlib.Path(args['out-path'])
    if not(args['modulation'] ^ args['distortion'] ^ args['delay']):
        raise argparse.ArgumentError(None, message="One and only one effect must be selected.")
    if args['modulation']:
        FX = pdb.Chorus()
        PARAM_RANGE = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
        PARAMS = ['p-rate_hz', 'p-depth', 'p-centre_delay_ms',
                'p-feedback', 'p-mix']
    elif args['delay']:
        FX = pdb.Delay()
        PARAM_RANGE = [(0, 1), (0, 1), (0, 1)]
        PARAMS = ['p-delay', 'p-feedback', 'p-mix']
    elif args['distortion']:
        FX = CustomDistortion()
        PARAM_RANGE = [(0, 60),
                       (50, 500), (-10, 10), (0.5, 2),
                       (500, 2000), (-10, 10), (0.5, 2)]
        PARAMS = ["p-drive_db",
                  "p-cutoff_frequency_hz_lo", "p-gain_db_lo", "p-q_lo",
                  "p-cutoff_frequency_hz_hi", "p-gain_db_hi", "p-q_hi"]
    else:
        raise NotImplementedError("You should not be here.")
    NUM_RUNS = args['num_runs']
    dataframe = pd.DataFrame(columns=PARAMS, dtype='float64')
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    if (OUT_PATH / "params.csv").exists():
        raise ValueError("Output directory already has a params.csv file. Aborting.")
    for i in tqdm(range(NUM_RUNS), position=1):
        for file in tqdm(DATA_PATH.rglob('*.wav'), total=1872, position=0, leave=True):
            # Generate random parameter values, rounded to the nearest hundredth
            params = [round(random.random(), 2) for p in range(len(PARAMS))]
            FX.set_fx_params(params, param_range=PARAM_RANGE, flat=True)
            audio, rate = util.read_audio(file, normalize=True, add_noise=False)
            processed = FX.process(audio, rate)
            dataframe.loc[file.stem + '_' + str(i)] = params
            if args['downsample']:
                audio = audio[0, ::2]
                rate = rate // 2
            processed = processed[0] / torch.max(torch.abs(processed[0]))
            sf.write(OUT_PATH / (file.stem + '_' + str(i) + file.suffix), processed, int(rate))
        dataframe.to_csv(OUT_PATH / "params.csv")
        dataframe.to_pickle(OUT_PATH / "params.pkl")
    with open(OUT_PATH / "config.txt", 'w') as f:
        f.write(str(PARAM_RANGE))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to process clean audio with randomly configured Fx.")
    parser.add_argument('--data-path', '-d', type=str,
                        help="Path to the clean sounds.")
    parser.add_argument('--out-path', '-o', type=str,
                        help="Path to store the produced files.")
    parser.add_argument('--downsample', '-D', action='store_true',
                        help="Flag to divide the sampling rate by 2.")
    parser.add_argument('--modulation', action='store_true',
                        help="Use the Modulation effect.")
    parser.add_argument('--delay', action='store_true',
                        help="Use the Delay effect.")
    parser.add_argument('--distortion', action='store_true',
                        help="Use the Distortion effect.")
    parser.add_argument('--num-runs', '-n', type=int, default=20,
                        help="Number of runs over the clean sounds. Default is 20.")
    sys.exit(main(parser))
