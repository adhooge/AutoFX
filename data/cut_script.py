import os

from tqdm import tqdm

import util
import pathlib
import soundfile as sf

IN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050")
OUT_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050_cut")

if not OUT_PATH.exists():
    os.mkdir(OUT_PATH)

for file in tqdm(IN_PATH.iterdir()):
    file = pathlib.Path(file)
    if file.suffix == '.wav':
        audio, rate = sf.read(file)
        cut_audio = util.cut2onset(audio, rate)
        sf.write(OUT_PATH / file.name, cut_audio, rate)
