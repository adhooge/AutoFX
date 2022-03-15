"""
Script for data augmentation
"""
import argparse
import pathlib
import sys
import pedalboard
import soundfile
from tqdm import tqdm
import util


def main(parser: argparse.ArgumentParser) -> int:
    args = vars(parser.parse_args())
    transforms = ''
    fx = []
    if args.get('chorus'):
        fx.append(pedalboard.Chorus())
        transforms += '_chorus'
    if args.get('distortion'):
        fx.append(pedalboard.Distortion())
        transforms += '_distortion'
    if args.get('reverb'):
        fx.append(pedalboard.Reverb())
        transforms += '_reverb'
    if len(fx) == 0:
        raise ValueError("Choose at least one effect to apply to the dry audio.")
    board = pedalboard.Pedalboard(fx)
    in_path = args['in_path']
    out_path = args['out_path']
    if not in_path.is_absolute():
        in_path = pathlib.Path.cwd() / in_path
    if not out_path.is_absolute():
        out_path = pathlib.Path.cwd() / out_path
    if out_path.exists():
        if not out_path.is_dir():
            raise NotADirectoryError("Output path is not a directory.")
        elif any(out_path.iterdir()) and not args['force']:
            raise FileExistsError("Output directory is not empty. Add --force or -f to overwrite anyway.")
    else:
        out_path.mkdir()
    if not in_path.exists():
        raise FileNotFoundError("Input directory cannot be found.")
    else:
        if not in_path.is_dir():
            raise NotADirectoryError("Input path is not a directory.")
        for file in tqdm(in_path.iterdir()):
            audio, rate = util.read_audio(file)
            audio = util.apply_fx(audio, rate, board)
            out_name = file.stem + transforms + '.wav'
            soundfile.write(out_path / out_name, audio, rate)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data augmentation by applying effects on dry sounds.")
    parser.add_argument('--chorus', action='store_true')
    parser.add_argument('--distortion', action='store_true')
    parser.add_argument('--reverb', action='store_true')
    parser.add_argument('--in-path', '-i', default=pathlib.PurePath('.'), type=pathlib.PurePath,
                        help="Path to the dry sounds. Defaults to current directory.")
    parser.add_argument('--out-path', '-o', default=pathlib.PurePath('./out'), type=pathlib.PurePath,
                        help="Where to store the processed files. Defaults to ./out")
    parser.add_argument('--force', '-f', action='store_true')
    sys.exit(main(parser))
