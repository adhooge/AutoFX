import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf
import features as Ft
import functional as Fc
from tqdm import tqdm
import pandas as pd


def main(parser):
    args = vars(parser.parse_args())
    in_path = pathlib.Path(args['input_path'])
    out_path = pathlib.Path(args['output_path'])
    out_csv = out_path / (args['name'] + '.csv')
    out_pkl = out_path / (args['name'] + '.pkl')
    df = pd.read_csv(in_path / "params.csv")
    df['f-phase_fft_max'] = np.nan
    df['f-phase_freq'] = np.nan
    df['f-rms_fft_max'] = np.nan
    df['f-rms_freq'] = np.nan
    if not in_path.is_absolute():
        in_path = pathlib.Path.cwd() / in_path
    if not out_path.is_absolute():
        out_path = pathlib.Path.cwd() / out_path
    if not args['force']:
        if out_csv.exists() or out_pkl.exists():
            raise FileExistsError("Output directory is not empty. Add --force or -f to overwrite anyway.")
    if not out_path.exists():
        out_path.mkdir()
    if not in_path.exists():
        raise FileNotFoundError("Input directory cannot be found.")
    else:
        if not in_path.is_dir():
            raise NotADirectoryError("Input path is not a directory.")
        for file in tqdm(in_path.iterdir()):
            file = pathlib.Path(file)
            if file.suffix == '.wav':
                audio, rate = sf.read(file)
                phase = Ft.phase_fmax(audio)
                rms = Ft.rms_energy(audio)
                phase_fft_max, phase_freq = Fc.fft_max(phase)
                rms_fft_max, rms_freq = Fc.fft_max(rms)
                features = [phase_fft_max, phase_freq/512, rms_fft_max, rms_freq/512]
                df.loc[df['Unnamed: 0'] == file.stem,
                       ['f-phase_fft_max', 'f-phase_freq',
                        'f-rms_fft_max', 'f-rms_freq']] = features
    df.to_csv(out_csv)
    df.to_pickle(out_pkl)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computing features of audio files for a chorus effect.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the processed sounds.")
    parser.add_argument('--output-path', '-o', type=str or pathlib.Path,
                        help="Path to an output folder to store the analysis results.")
    parser.add_argument('--name', '-n', default='data', type=str,
                        help="Name to give to the output file")
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
