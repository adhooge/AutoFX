"""
Functions to extract the relevant features,
as mentioned in Stein et al. Automatic Detection of Audio effects [...] 2010.
References:
    [1]: Geoffroy Peeters, Technical report of the CUIDADO project, 2003
    [2]: Tae Hong Park, Towards automatic musical instrument timbre recognition, 2004 (PhD thesis)
"""

import numpy as np


def _geom_mean(arr: np.ndarray):
    """
    Compute the geometrical mean of an array through log conversion to avoid overflow.
    """
    return np.exp(np.mean(np.log(arr)))


def spectral_centroid(*, mag: np.ndarray = None, stft: np.ndarray = None, freq: np.ndarray = None) -> np.ndarray:
    """
    Spectral centroid of each frame.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix of the short time fourier transform;
    :param freq: frequency of each frequency bin, in Hertz;
    :return: spectral centroid of each input frame.
    """
    if mag is None:
        mag = np.abs(stft)
    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=1)
    norm_mag = mag/np.sum(mag, axis=0)
    cent = np.sum(norm_mag * freq[:, np.newaxis], axis=0)
    return cent


def spectral_spread(*, mag: np.ndarray = None, stft: np.ndarray = None,
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
        mag = np.abs(stft)
    if cent is None:
        cent = spectral_centroid(mag=mag, freq=freq)
    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=1)
    spread = np.zeros_like(cent)
    norm_mag = mag/np.sum(mag, axis=0)
    for (i, centroid) in enumerate(cent):
        cnt_freq = freq - centroid
        spread[i] = np.sum(norm_mag[:, i] * np.square(cnt_freq))
    return spread


def spectral_skewness(*, mag: np.ndarray = None, stft: np.ndarray = None,
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
        mag = np.abs(stft)
    if cent is None:
        cent = spectral_centroid(mag=mag, freq=freq)
    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=1)
    skew = np.zeros_like(cent)
    norm_mag = mag/np.sum(mag, axis=0)
    for (i, centroid) in enumerate(cent):
        cnt_freq = freq - centroid
        skew[i] = np.sum(norm_mag[:, i] * np.power(cnt_freq, 3))
    return skew


def spectral_kurtosis(*, mag: np.ndarray = None, stft: np.ndarray = None,
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
        mag = np.abs(stft)
    if cent is None:
        cent = spectral_centroid(mag=mag, freq=freq)
    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=1)
    kurt = np.zeros_like(cent)
    norm_mag = mag/np.sum(mag, axis=0)
    for (i, centroid) in enumerate(cent):
        cnt_freq = freq - centroid
        kurt[i] = np.sum(norm_mag[:, i] * np.power(cnt_freq, 4))
    return kurt


def spectral_flux(mag: np.ndarray, q_norm: int = 1):
    """
    Amount of frame-to-frame fluctuation in time.
    See: Tae Hong Park, Towards automatic musical instrument timbre recognition, 2004 (PhD thesis)

    :param mag: (num_frames, N_fft) Matrix of the frame-by-frame magnitude;
    :param q_norm: order of the q norm to use. Defaults to 1;
    :return flux: (num_frames, 1) matrix of the spectral flux, set to 0 for the first frame.
    """
    if mag.ndim == 1:
        raise ValueError("Spectral flux cannot be computed on a single frame.")
    num_frames = mag.shape[1]
    flux = np.zeros((num_frames, 1))
    for fr in range(1, num_frames):
        diff = mag[:, fr] - mag[:, fr - 1]
        flux[fr] = (np.sum(np.power(np.abs(diff), q_norm)))**(1/q_norm)
    return flux


def spectral_rolloff(mag: np.ndarray, threshold: float = 0.95, freq: np.ndarray = None):
    """
    The spectral roll-off point is the frequency so that threshold% of the signal energy is contained
    below that frequency.
    See: Geoffroy Peeter, Technical report of the CUIDADO project, 2004.

    :param mag: (num_frames, N_fft) Matrix of the frame-by-frame magnitude;
    :param threshold: Ratio of the signal energy to use. Defaults to 0.95;
    :param freq: array of the frequency in Hertz of each frequency bin. If None, result is given as a bin number;
    :return rolloff: (num_frames, 1) matrix of the spectral roll-off for each frame, in Hz or #bin.
    """
    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=1)
    num_frames = mag.shape[1]
    energy = np.power(mag, 2)
    rolloff = np.empty((num_frames, 1))
    for fr in range(num_frames):
        flag = True
        tot_energy = np.sum(energy[:, fr])
        cumul_energy = np.cumsum(energy[:, fr])
        for (bin_num, ener) in enumerate(cumul_energy):
            if ener > threshold * tot_energy and flag:
                flag = False
                if freq is None:
                    rolloff[fr] = bin_num
                else:
                    rolloff[fr] = freq[bin_num]
            elif not flag:
                break
    return rolloff


def spectral_slope(mag: np.ndarray, freq: np.ndarray = None):
    """
    The spectral slope represents the amount of decreasing of the spectral amplitude [1].

    :param mag: (num_frames, N_fft) matrix of the frame_by_frame magnitude;
    :param freq: array of the frequency in Hertz of each frequency bin. If None, result is given in bins;
    :return slope: (num_frames, 1) array of the frame-wise spectral slope, in Hz or bins depending on freq.
    """
    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=1)
    n_fft, num_frames = mag.shape
    if freq is None:
        freq = np.arange(n_fft)
    slope = np.empty((num_frames, 1))
    for fr in range(num_frames):
        num = (n_fft * np.sum(np.multiply(freq, mag[:, fr])) - np.sum(freq)*np.sum(mag[:, fr]))
        denom = n_fft * np.sum(np.power(freq, 2)) - np.sum(freq)**2
        slope[fr] = num/denom
    return slope


def spectral_flatness(mag: np.ndarray, bands: int or list = 1, rate: float = None):
    """
    The spectral flatness is a measure of the noisiness of a spectrum. [1, 2]

    :param mag: (num_frames, N_fft) matrix of the frame_by_frame magnitude;
    :param bands: Default=1. If int: number of frequency bands to consider, regularly spaced in the spectrum.
                             If list: List of (start, end) for each frequency band. Should be in Hz if rate is set,
                             # bins otherwise;
    :param rate: sampling rate of the signal in Hertz;
    :return flatness: (num_frames, bands) matrix of the spectral flatness of each frame and frequency band.
    """
    def freq2bin(freq, rate, n_fft):
        return int(freq*2*n_fft/rate)
    if mag.ndim == 1:
        mag = np.expand_dims(mag, axis=1)
    n_fft, num_frames = mag.shape
    if not isinstance(bands, (int, list)):
        raise TypeError("Frequency bands should be an integer number or a list of Tuples.")
    if isinstance(bands, int):
        if bands == 1:
            bands = [(0, n_fft - 1)]
        else:
            tmp = []
            start = 0
            band_size = n_fft // bands
            for band in range(bands):
                tmp.append((start, start+band_size))
                start += band_size
    elif isinstance(bands, list):
        if rate is not None:
            bands = list(map(lambda x: (freq2bin(x[0], rate, n_fft), freq2bin(x[1], rate, n_fft)), bands))
    flatness = np.empty((num_frames, len(bands)))
    for fr in range(num_frames):
        for (b, band) in enumerate(bands):
            arr = mag[band[0]:band[1], fr]
            flatness[fr, b] = _geom_mean(arr)/np.mean(arr)
    return flatness
