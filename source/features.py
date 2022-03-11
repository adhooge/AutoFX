"""
Functions to extract the relevant features,
as mentioned in Stein et al. Automatic Detection of Audio effects [...] 2010.
"""

import librosa
import numpy as np


def spectral_centroid(*, mag: np.ndarray = None, stft: np.ndarray = None, freq: np.ndarray = None) -> np.ndarray:
    """
    Spectral centroid of each frame.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix of the short time fourier transform;
    :param freq: frequency of each frequency bin, in Hertz;
    :return: spectral centroid of each input frame.
    """
    if mag is None:
        mag, _ = librosa.magphase(stft)
    norm_mag = mag/np.sum(mag, axis=0)
    cent = np.sum(np.multiply(norm_mag, freq[:, np.newaxis]), axis=0)
    return cent


def spectral_spread(mag: np.ndarray = None, stft: np.ndarray = None,
                    cent: np.ndarray = None, freq: np.ndarray = None) -> np.ndarray:
    """
    Spectral spread of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :return spread: spectral spread of each input frame.
    """
    if mag is None:
        mag, _ = librosa.magphase(stft)
    if cent is None:
        cent = spectral_centroid(mag=mag, freq=freq)
    spread = np.zeros_like(cent)
    norm_mag = mag/np.sum(mag, axis=0)
    for (i, centroid) in enumerate(cent):
        cnt_freq = freq - centroid
        spread[i] = np.sum(np.multiply(norm_mag[:, i], np.square(cnt_freq)))
    return spread


def spectral_skewness(mag: np.ndarray = None, stft: np.ndarray = None,
                    cent: np.ndarray = None, freq: np.ndarray = None) -> np.ndarray:
    """
    Spectral skewness of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :return skew: spectral skewness of each input frame.
    """
    if mag is None:
        mag, _ = librosa.magphase(stft)
    if cent is None:
        cent = spectral_centroid(mag=mag, freq=freq)
    skew = np.zeros_like(cent)
    norm_mag = mag/np.sum(mag, axis=0)
    for (i, centroid) in enumerate(cent):
        cnt_freq = freq - centroid
        skew[i] = np.sum(np.multiply(norm_mag[:, i], np.power(cnt_freq, 3)))
    return skew


def spectral_kurtosis(mag: np.ndarray = None, stft: np.ndarray = None,
                    cent: np.ndarray = None, freq: np.ndarray = None) -> np.ndarray:
    """
    Spectral kurtosis of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :return kurt: spectral kurtosis of each input frame.
    """
    if mag is None:
        mag, _ = librosa.magphase(stft)
    if cent is None:
        cent = spectral_centroid(mag=mag, freq=freq)
    kurt = np.zeros_like(cent)
    norm_mag = mag/np.sum(mag, axis=0)
    for (i, centroid) in enumerate(cent):
        cnt_freq = freq - centroid
        kurt[i] = np.sum(np.multiply(norm_mag[:, i], np.power(cnt_freq, 4)))
    return kurt


def spectral_flux():
    return NotImplemented


def spectral_rolloff():
    return NotImplemented


def spectral_slope():
    return NotImplemented


def spectral_flatness():
    return NotImplemented
