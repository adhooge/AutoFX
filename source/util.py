"""
Utility functions for data processing.
"""

import pedalboard as pdb
import soundfile as sf
from numpy import ndarray


def apply_fx(audio, rate: float, board: pdb.Pedalboard):
    """
    Apply effects to audio.
    :param audio: Array representing the audio to process;
    :param rate: Sample rate of the audio to process;
    :param board: Pedalboard instance of FX to apply. Can contain one or several FX;
    :return: Processed audio.
    """
    return board.process(audio, rate)


def read_audio(path: str, **kwargs) -> ndarray:
    """
    Wrapper function to read an audio file using soundfile.read
    :param path: Path to audio file to read;
    :param kwargs: Keyword arguments to pass to soundfile.read;
    :return audio, rate: Read audio and the corresponding sample rate.
    """
    audio, rate = sf.read(path, **kwargs)
    return audio, rate

