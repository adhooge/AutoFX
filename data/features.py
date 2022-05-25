"""
Functions to extract the relevant features,
as mentioned in Stein et al. Automatic Detection of Audio effects [...] 2010.
References:
    [1]: Geoffroy Peeters, Technical report of the CUIDADO project, 2003
    [2]: Tae Hong Park, Towards automatic musical instrument timbre recognition, 2004 (PhD thesis)
"""
import librosa
import numpy as np
import torch
import torchaudio.functional
from torch import Tensor


def _geom_mean(arr: np.ndarray):
    """
    Compute the geometrical mean of an array through log conversion to avoid overflow.
    """
    return np.exp(np.mean(np.log(arr)))


def spectral_centroid(*, mag: np.ndarray or Tensor = None, stft: np.ndarray or Tensor = None, freq: np.ndarray = None,
                      torch_compat: bool = False) -> np.ndarray or Tensor:
    """
    Spectral centroid of each frame.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix of the short time fourier transform;
    :param freq: frequency of each frequency bin, in Hertz;
    :param torch_compat: enable torch-compatible computation. Default is False
    :return: spectral centroid of each input frame.
    """
    if torch_compat:
        if mag is None:
            mag = torch.abs(stft)
        batch_size = mag.shape[0]
        if freq is None:
            freq = torch.linspace(0, 0.5, mag.shape[-1])
            freq = torch.vstack([freq]*batch_size)
        norm_mag = mag / torch.sum(mag, dim=-1, keepdim=True)
        cent = torch.sum(norm_mag * freq, dim=-1, keepdim=True)
        return cent
    else:
        if mag is None:
            mag = np.abs(stft)
        if freq is None:
            freq = np.linspace(0, 0.5, mag.shape[-1])
        if mag.ndim == 1:
            mag = np.expand_dims(mag, axis=1)
        norm_mag = mag / np.sum(mag, axis=0)
        cent = np.sum(norm_mag.T * freq, axis=1)
        return cent


def spectral_spread(*, mag: np.ndarray or Tensor = None, stft: np.ndarray or Tensor = None,
                    cent: np.ndarray = None, freq: np.ndarray = None,
                    torch_compat: bool = False) -> np.ndarray or Tensor:
    """
    Spectral spread of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :param torch_compat: enable torch-compatible computation. Default is False;
    :return spread: spectral spread of each input frame.
    """
    if torch_compat:
        if mag is None:
            mag = torch.abs(stft)
        batch_size = mag.shape[0]
        if cent is None:
            cent = spectral_centroid(mag=mag, freq=freq, torch_compat=True)
        if freq is None:
            freq = torch.linspace(0, 0.5, mag.shape[-1])
            freq = torch.vstack([freq]*batch_size)
        norm_mag = mag / torch.sum(mag, dim=-1, keepdim=True)
        cnt_freq = freq - cent
        spread = torch.sum(norm_mag * torch.square(cnt_freq), dim=-1, keepdim=True)
        return spread
    else:
        if mag is None:
            mag = np.abs(stft)
        if cent is None:
            cent = spectral_centroid(mag=mag, freq=freq)
        if freq is None:
            freq = np.linspace(0, 0.5, mag.shape[-1])
        if mag.ndim == 1:
            mag = np.expand_dims(mag, axis=1)
        norm_mag = mag / np.sum(mag, axis=0)
        cnt_freq = freq - cent
        spread = np.sum(np.multiply(norm_mag.T, np.square(cnt_freq)), axis=1)
        return spread


def spectral_skewness(*, mag: np.ndarray or Tensor = None,
                      stft: np.ndarray or Tensor = None,
                      cent: np.ndarray = None, freq: np.ndarray = None,
                      torch_compat: bool = False) -> np.ndarray or Tensor:
    """
    Spectral skewness of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :param torch_compat: enable torch-compatible computation. Default is False;
    :return skew: spectral skewness of each input frame.
    """
    if torch_compat:
        if mag is None:
            mag = torch.abs(stft)
        batch_size = mag.shape[0]
        if cent is None:
            cent = spectral_centroid(mag=mag, freq=freq, torch_compat=True)
        if freq is None:
            freq = torch.linspace(0, 0.5, mag.shape[-1])
            freq = torch.vstack([freq]*batch_size)
        norm_mag = mag / torch.sum(mag, dim=-1, keepdim=True)
        cnt_freq = freq - cent
        skew = torch.sum(norm_mag * torch.pow(cnt_freq, 3), dim=-1, keepdim=True)
        return skew
    else:
        if mag is None:
            mag = np.abs(stft)
        if cent is None:
            cent = spectral_centroid(mag=mag, freq=freq)
        if freq is None:
            freq = np.linspace(0, 0.5, mag.shape[-1])
        if mag.ndim == 1:
            mag = np.expand_dims(mag, axis=1)
        norm_mag = mag / np.sum(mag, axis=0)
        cnt_freq = freq - cent
        skew = np.sum(norm_mag.T * np.power(cnt_freq, 3), axis=1)
        return skew


def spectral_kurtosis(*, mag: np.ndarray or Tensor = None, stft: np.ndarray or Tensor = None,
                      cent: np.ndarray = None, freq: np.ndarray = None,
                      torch_compat: bool = False) -> np.ndarray or Tensor:
    """
    Spectral kurtosis of each frame of the input signal.

    See: Geoffroy Peeters, Cuidado project Technical report, 2003.

    :param mag: Magnitude spectrogram of the input signal;
    :param stft: Complex matrix representing a Short Time Fourier Transform;
    :param cent: Array of the spectral centroid of each frame;
    :param freq: frequency of each frequency bin, in Hertz;
    :param torch_compat: enable torch-compatible computation. Default is False;
    :return kurt: spectral kurtosis of each input frame.
    """
    if torch_compat:
        if mag is None:
            mag = torch.abs(stft)
        batch_size = mag.shape[0]
        if cent is None:
            cent = spectral_centroid(mag=mag, freq=freq, torch_compat=True)
        if freq is None:
            freq = torch.linspace(0, 0.5, mag.shape[-1])
            freq = torch.vstack([freq]*batch_size)
        norm_mag = mag / torch.sum(mag, dim=-1, keepdim=True)
        cnt_freq = freq - cent
        kurt = torch.sum(norm_mag * torch.pow(cnt_freq, 4), dim=-1, keepdim=True)
        return kurt
    else:
        if mag is None:
            mag = np.abs(stft)
        if cent is None:
            cent = spectral_centroid(mag=mag, freq=freq)
        if freq is None:
            freq = np.linspace(0, 0.5, mag.shape[-1])
        if mag.ndim == 1:
            mag = np.expand_dims(mag, axis=1)
        norm_mag = mag / np.sum(mag, axis=0)
        cnt_freq = freq - cent
        kurt = np.sum(norm_mag.T * np.power(cnt_freq, 4), axis=1)
        return kurt


def spectral_flux(mag: np.ndarray or Tensor, q_norm: int = 1,
                  torch_compat: bool = True) -> np.ndarray or Tensor:
    """
    Amount of frame-to-frame fluctuation in time.
    See: Tae Hong Park, Towards automatic musical instrument timbre recognition, 2004 (PhD thesis)

    :param mag: (N_fft, num_frames) Matrix of the frame-by-frame magnitude;
    :param q_norm: order of the q norm to use. Defaults to 1;
    :param torch_compat: enable torch-compatible computation. Default is False;
    :return flux: (1, num_frames) matrix of the spectral flux, set to 0 for the first frame.
    """
    if torch_compat:
        diff = torch.diff(mag, n=1, dim=-1)
        flux = torch.pow(torch.sum(torch.pow(torch.abs(diff), q_norm)), 1/q_norm)
        return flux
    else:
        if mag.ndim == 1:
            raise ValueError("Spectral flux cannot be computed on a single frame.")
        num_frames = mag.shape[1]
        flux = np.zeros((num_frames, 1))
        for fr in range(1, num_frames):
            diff = mag[:, fr] - mag[:, fr - 1]
            flux[fr] = (np.sum(np.power(np.abs(diff), q_norm))) ** (1 / q_norm)
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
        num = (n_fft * np.sum(np.multiply(freq, mag[:, fr])) - np.sum(freq) * np.sum(mag[:, fr]))
        denom = n_fft * np.sum(np.power(freq, 2)) - np.sum(freq) ** 2
        slope[fr] = num / denom
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
        return int(freq * 2 * n_fft / rate)

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
                tmp.append((start, start + band_size))
                start += band_size
    elif isinstance(bands, list):
        if rate is not None:
            bands = list(map(lambda x: (freq2bin(x[0], rate, n_fft), freq2bin(x[1], rate, n_fft)), bands))
    flatness = np.empty((num_frames, len(bands)))
    for fr in range(num_frames):
        for (b, band) in enumerate(bands):
            arr = mag[band[0]:band[1], fr]
            flatness[fr, b] = _geom_mean(arr) / np.mean(arr)
    return flatness


def get_mfcc(audio, rate, num_coeff):
    mfcc = librosa.feature.mfcc(y=audio, sr=rate)
    return mfcc[:num_coeff]


def phase_fmax_batch(audio):
    # TODO: TEST
    """
    Copy of phase_fmax using torch tools for batch processing and GPU compatible processing.
    :param audio:
    :return:
    """
    transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=256)
    stft = transform(audio)
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    spec_sum = torch.sum(mag, dim=-1)
    max_bins = torch.argmax(spec_sum, dim=-1)
    phase_freq_max = phase[max_bins]

    mag_max_bin_mask = mag[max_bins]
    thresh = torch.max(mag_max_bin_mask, dim=-1) / 8
    phase_freq_max = torch.where(mag_max_bin_mask > thresh, phase_freq_max, 0)
    phase_freq_max_t = torch.nonzero(phase_freq_max)

    # unwrap phase
    phase_fmax_straight_t = torch.clone(phase_freq_max_t)
    diff_mean_sign = torch.mean(torch.sign(torch.diff(phase_freq_max_t)))
    phase_correction = torch.zeros_like(diff_mean_sign)
    batch_size = audio.shape[0]
    for b in range(batch_size):
        if diff_mean_sign[b] > 0:
            pass # TODO: implement without for loop
    raise NotImplementedError

def phase_fmax(sig):
    """
    Copied from:
    https://github.com/henrikjuergens/guitar-fx-extraction/blob/442adab577a090e27de12d779a6d8a0aa917fe1f/featextr.py#L63
    Analyses phase error of frequency bin with maximal amplitude
        compared to pure sine wave"""
    D = librosa.stft(y=sig, hop_length=256)[20:256]
    S, P = librosa.core.magphase(D)
    phase = np.angle(P)
    # plots.phase_spectrogram(phase)

    spec_sum = S.sum(axis=1)
    max_bin = spec_sum.argmax()
    phase_freq_max = phase[max_bin]
    # plots.phase_fmax(phase_freq_max)

    S_max_bin_mask = S[max_bin]
    thresh = S[max_bin].max() / 8
    phase_freq_max = np.where(S_max_bin_mask > thresh, phase_freq_max, 0)
    phase_freq_max_t = np.trim_zeros(phase_freq_max)  # Using only phase with strong signal

    # unwrap phase
    phase_fmax_straight_t = np.copy(phase_freq_max_t)
    diff_mean_sign = np.mean(np.sign(np.diff(phase_freq_max_t)))
    if diff_mean_sign > 0:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i - 1]) > np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] += 2 * np.pi
    else:
        for i in range(1, len(phase_fmax_straight_t)):
            if np.sign(phase_freq_max_t[i - 1]) < np.sign(phase_freq_max_t[i]):
                phase_fmax_straight_t[i:] -= 2 * np.pi

    x_axis_t = np.arange(0, len(phase_fmax_straight_t))
    coeff = np.polyfit(x_axis_t, phase_fmax_straight_t, 1)
    linregerr_t = np.copy(phase_fmax_straight_t)
    linregerr_t -= (coeff[0] * x_axis_t + coeff[1])
    linregerr_t = np.reshape(linregerr_t, (1, len(linregerr_t)))
    # plots.phase_error_unwrapped(phase_fmax_straight_t, coeff, x_axis_t)
    return linregerr_t


def pitch_curve(audio, rate, fmin, fmax,
                default_f: float = None, torch_compat: bool=False):
    """
    fmin and fmax need to be quite far from one another for the algorithm to work
    :param audio:
    :param rate:
    :param fmin:
    :param fmax:
    :return:
    """
    if audio.ndim == 3 or torch_compat:     # batch treatment
        f0 = torchaudio.functional.detect_pitch_frequency(audio, rate)
        return f0
    else:
        f0 = librosa.pyin(y=audio, fmin=int(fmin), fmax=int(fmax), sr=rate, fill_na=default_f)
        return f0[0]


def rms_energy(audio):
    return librosa.feature.rms(y=audio)