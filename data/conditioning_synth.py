import argparse

import torch
import torch.nn.functional as F
import pandas as pd
import pathlib
import sys

import torchaudio
import tqdm


def main(parser):
    args = vars(parser.parse_args())
    input_path = pathlib.Path(args['input_path'])
    name = args['name']
    if args['append']:
        df = pd.read_csv((input_path / name).with_suffix('.csv'), index_col=0)
        df['conditioning'] = None
    else:
        df = pd.DataFrame(columns=['conditioning'])
    clf = torch.jit.load(args['model'])
    for file in tqdm.tqdm(input_path.rglob('*.wav')):
        audio, rate = torchaudio.load(input_path / file)
        if audio.shape[-1] < 44100 and args['padding']:
            to_pad = 44100 - audio.shape[-1]
            audio = F.pad(audio, (to_pad, 0))
        conditioning = clf(audio)
        conditioning = conditioning.detach().numpy() / 10
        df.loc[df["Unnamed: 0"] == file.stem, 'conditioning'] = conditioning
    df.to_csv(input_path / 'data.csv')
    df.to_pickle(input_path / 'data.pkl')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Class conditioning.")
    parser.add_argument('--input-path', '-i', type=str or pathlib.Path,
                        help="Path to the processed sounds.")
    parser.add_argument('--model', '-m', type=str or pathlib.Path,
                        help="Path to compiled classifier.")
    parser.add_argument('--name', '-n', default='data', type=str,
                        help="Name to give to the output file.")
    parser.add_argument('--append', '-a', action='store_true')
    parser.add_argument('--padding', '-p', action='store_true')
    sys.exit(main(parser))
