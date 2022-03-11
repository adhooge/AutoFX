"""
Utility functions for data processing.
"""
from typing import Tuple, Any

import pedalboard as pdb
import soundfile as sf
import numpy as np
import librosa


def apply_fx(audio, rate: float, board: pdb.Pedalboard):
    """
    Apply effects to audio.
    :param audio: Array representing the audio to process;
    :param rate: Sample rate of the audio to process;
    :param board: Pedalboard instance of FX to apply. Can contain one or several FX;
    :return: Processed audio.
    """
    return board.process(audio, rate)


def read_audio(path: str, **kwargs) -> Tuple[np.ndarray, float]:
    """
    Wrapper function to read an audio file using soundfile.read
    :param path: Path to audio file to read;
    :param kwargs: Keyword arguments to pass to soundfile.read;
    :return audio, rate: Read audio and the corresponding sample rate.
    """
    audio, rate = sf.read(path, **kwargs)
    return audio, rate


def energy_envelope(audio: np.ndarray, rate: float, window_size: float = 100, method: str = 'rms') -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Compute the energy envelope of a signal according to the selected method. A default window size
    of 100ms is used for a 5Hz low-pass filtering (see Peeters' Cuidado Project report, 2003).

    Valid methods are Root Mean Square ('rms') and Absolute ('abs').

    :param audio: Array representing the signal to be analysed;
    :param rate: sampling rate of the signal;
    :param window_size: Window size in ms for low-pass filtering. Default: 100;
    :param method: name of the method to use. Default: 'rms'.
    :return (env, times): energy envelope of the signal and the corresponding times in seconds.
    """
    if method == 'rms':
        window_size_n = window_size * rate / 1000
        chunks = np.ceil(audio.size / window_size_n)
        env = list(map(lambda x: np.sqrt(np.mean(np.square(x))), np.array_split(audio, chunks)))
        times = np.arange(chunks) * window_size / 1000
        return np.array(env), times
    elif method == 'abs':
        window_size_n = window_size * rate / 1000
        chunks = np.ceil(audio.size / window_size_n)
        env = list(map(lambda x: np.mean(np.abs(x)), np.array_split(audio, chunks)))
        times = np.arange(chunks) * window_size / 1000
        return np.array(env), times
    else:
        raise NotImplementedError


def find_attack(energy: np.ndarray, method: str,
                start_threshold: float = None, end_threshold: float = None,
                times: np.ndarray = None) -> Tuple[float, float]:
    """
    Find beginning and end of attack from energy envelope using a fixed or adaptive threshold method.

    See: Geoffroy Peeters, A large set of audio features for sound description in the CUIDADO project, 2003.


    :param energy: np.ndarray of the energy envelope;
    :param method: 'adaptive' or 'fixed';
    :param end_threshold: max energy ratio to detect end of attack in 'fixed' method;
    :param start_threshold: max energy ratio to detect start of attack in 'fixed' method;
    :param times: timestamps in seconds corresponding to the energy samples. If set, start and end are given in seconds.
    :return (start, end): indices from energy to the corresponding instants.
    """
    if method == 'fixed':
        if start_threshold is None or end_threshold is None:
            raise ValueError("For 'fixed' method, start and end thresholds have to be set.")
        max_energy = np.max(energy)
        max_pos = np.argmax(energy)
        start, end = None, None
        for i in range(max_pos):
            if energy[i] >= max_energy * start_threshold and start is None:
                start = i
            if energy[i] >= max_energy * end_threshold and end is None:
                end = i
        if times is not None:
            start = times[start]
            end = times[end]
        return start, end
    elif method == 'adaptive':
        raise NotImplementedError("Will be added if necessary")
        # if start_threshold is not None or end_threshold is not None:
        #    raise UserWarning("Setting thresholds is useless in 'adaptive' method. Values are ignored.")
    else:
        raise NotImplementedError("method should be 'fixed' or 'adaptive'")


def get_stft(audio: np.ndarray, rate: float, fft_size: int, hop_size: int = None, window: Any = 'hann',
             window_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function to obtain the Short Time Fourier Transform of a sound. As of now, simply calls Librosa.

    :param audio: np.ndarray of the sound to analyse;
    :param rate: sampling rate of the audio signal. Necessary to return the correct frequencies;
    :param fft_size: Size of the fft frames in samples. If fft_size > window_size, the windowed signal is zero padded;
    :param hop_size: hop size between frames in samples;
    :param window: window to use, can be a string, function, array...
    :param window_size: size of the window in samples.
    :return stft: Complex-valued matrix of short-term Fourier transform coefficients.
    :return freq: array of the frequency bins values in Hertz.
    """
    stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, win_length=window_size, window=window)
    freq = librosa.fft_frequencies(sr=rate, n_fft=fft_size)
    return stft, freq
