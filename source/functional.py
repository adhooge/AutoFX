"""
Functionals to be used for simpler representation of time changing-features.
"""
import librosa.feature
import numpy as np
from numpy.typing import ArrayLike
from config import DATA_DICT
from scipy.stats import skew, kurtosis
import util


def f_max(arr: ArrayLike) -> float:
    return np.max(arr)


def f_min(arr: ArrayLike) -> float:
    return np.min(arr)


def f_avg(arr: ArrayLike):
    return np.mean(arr)


def f_std(arr: ArrayLike):
    return np.std(arr)


def f_skew(arr: ArrayLike):
    return skew(arr)


def f_kurt(arr: ArrayLike):
    return kurtosis(arr)


def linear_regression(feat):
    lin_coeff, lin_residual, _ = np.polyfit(np.arange(len(feat)), feat, 1, full=True)
    return lin_coeff, lin_residual


def quad_reg(feat):
    quad_coeff, quad_residual, _ = np.polyfit(np.arange(len(feat)), feat, 2, full=True)
    return quad_coeff, quad_residual


def fft_max(feat):
    """
    https://github.com/henrikjuergens/guitar-fx-extraction/blob/master/featextr.py
    :param feat:
    :return:
    """
    dc_feat = feat - np.mean(feat)
    dc_feat_w = dc_feat * np.hanning(len(dc_feat))
    rfft = np.fft.rfft(dc_feat_w, 1024)
    rfft = np.abs(rfft) * 4 / 1024
    rfft[:16] = np.zeros(16)    # TODO: Find why?
    rfft_max = np.max(rfft)
    return rfft_max


def estim_derivative(feat, **kwargs):
    return librosa.feature.delta(feat, **kwargs)


def feat_vector(feat: dict, pitch: float) -> dict:
    """
    Returns a dict of all the functionals listed in config.DATA_DICT from feat, a dictionary representing the
    spectral features attribute of a SoundSample object.
    :param feat: Dictionary of the features;
    :param pitch: Pitch in Hertz of the note. Necessary for normalization of some features.
    """
    out = DATA_DICT
    cent = feat['centroid']
    out['cent_avg'] = f_avg(cent)
    out['cent_std'] = f_std(cent)
    out['cent_skw'] = f_skew(cent)
    out['cent_krt'] = f_kurt(cent)
    out['cent_min'] = f_min(cent)
    out['cent_max'] = f_max(cent)
    spread = feat['spread']
    out['spread_avg'] = f_avg(spread)
    out['spread_std'] = f_std(spread)
    out['spread_skw'] = f_skew(spread)
    out['spread_krt'] = f_kurt(spread)
    out['spread_min'] = f_min(spread)
    out['spread_max'] = f_max(spread)
    skew = feat['skewness']
    out['skew_avg'] = f_avg(skew)
    out['skew_std'] = f_std(skew)
    out['skew_skw'] = f_skew(skew)
    out['skew_krt'] = f_kurt(skew)
    out['skew_min'] = f_min(skew)
    out['skew_max'] = f_max(skew)
    kurt = feat['kurtosis']
    out['kurt_avg'] = f_avg(kurt)
    out['kurt_std'] = f_std(kurt)
    out['kurt_skw'] = f_skew(kurt)
    out['kurt_krt'] = f_kurt(kurt)
    out['kurt_min'] = f_min(kurt)
    out['kurt_max'] = f_max(kurt)
    # pitch normalized
    n_cent = cent / pitch
    out['n_cent_avg'] = f_avg(n_cent)
    out['n_cent_std'] = f_std(n_cent)
    out['n_cent_skw'] = f_skew(n_cent)
    out['n_cent_krt'] = f_kurt(n_cent)
    out['n_cent_min'] = f_min(n_cent)
    out['n_cent_max'] = f_max(n_cent)
    n_spread = spread / pitch
    out['n_spread_avg'] = f_avg(n_spread)
    out['n_spread_std'] = f_std(n_spread)
    out['n_spread_skw'] = f_skew(n_spread)
    out['n_spread_krt'] = f_kurt(n_spread)
    out['n_spread_min'] = f_min(n_spread)
    out['n_spread_max'] = f_max(n_spread)
    n_skew = skew / pitch
    out['n_skew_avg'] = f_avg(n_skew)
    out['n_skew_std'] = f_std(n_skew)
    out['n_skew_skw'] = f_skew(n_skew)
    out['n_skew_krt'] = f_kurt(n_skew)
    out['n_skew_min'] = f_min(n_skew)
    out['n_skew_max'] = f_max(n_skew)
    n_kurt = kurt / pitch
    out['n_kurt_avg'] = f_avg(n_kurt)
    out['n_kurt_std'] = f_std(n_kurt)
    out['n_kurt_skw'] = f_skew(n_kurt)
    out['n_kurt_krt'] = f_kurt(n_kurt)
    out['n_kurt_min'] = f_min(n_kurt)
    out['n_kurt_max'] = f_max(n_kurt)
    flux = feat['flux']
    out['flux_avg'] = f_avg(flux)
    out['flux_std'] = f_std(flux)
    out['flux_skw'] = f_skew(flux)[0]
    out['flux_krt'] = f_kurt(flux)[0]
    out['flux_min'] = f_min(flux)
    out['flux_max'] = f_max(flux)
    rolloff = feat['rolloff']
    out['rolloff_avg'] = f_avg(rolloff)
    out['rolloff_std'] = f_std(rolloff)
    out['rolloff_skw'] = f_skew(rolloff)[0]
    out['rolloff_krt'] = f_kurt(rolloff)[0]
    out['rolloff_min'] = f_min(rolloff)
    out['rolloff_max'] = f_max(rolloff)
    slope = feat['slope']
    out['slope_avg'] = f_avg(slope)
    out['slope_std'] = f_std(slope)
    out['slope_skw'] = f_skew(slope)[0]
    out['slope_krt'] = f_kurt(slope)[0]
    out['slope_min'] = f_min(slope)
    out['slope_max'] = f_max(slope)
    flat = feat['flatness']
    out['flat_avg'] = f_avg(flat)
    out['flat_std'] = f_std(flat)
    out['flat_skw'] = f_skew(flat)[0]
    out['flat_krt'] = f_kurt(flat)[0]
    out['flat_min'] = f_min(flat)
    out['flat_max'] = f_max(flat)
    # hi-passed aka delta features
    cent_hp = util.hi_pass(cent)
    out['cent_hp_avg'] = f_avg(cent_hp)
    out['cent_hp_std'] = f_std(cent_hp)
    out['cent_hp_skw'] = f_skew(cent_hp)
    out['cent_hp_krt'] = f_kurt(cent_hp)
    out['cent_hp_min'] = f_min(cent_hp)
    out['cent_hp_max'] = f_max(cent_hp)
    spread_hp = util.hi_pass(spread)
    out['spread_hp_avg'] = f_avg(spread_hp)
    out['spread_hp_std'] = f_std(spread_hp)
    out['spread_hp_skw'] = f_skew(spread_hp)
    out['spread_hp_krt'] = f_kurt(spread_hp)
    out['spread_hp_min'] = f_min(spread_hp)
    out['spread_hp_max'] = f_max(spread_hp)
    skew_hp = util.hi_pass(skew)
    out['skew_hp_avg'] = f_avg(skew_hp)
    out['skew_hp_std'] = f_std(skew_hp)
    out['skew_hp_skw'] = f_skew(skew_hp)
    out['skew_hp_krt'] = f_kurt(skew_hp)
    out['skew_hp_min'] = f_min(skew_hp)
    out['skew_hp_max'] = f_max(skew_hp)
    kurt_hp = util.hi_pass(kurt)
    out['kurt_hp_avg'] = f_avg(kurt_hp)
    out['kurt_hp_std'] = f_std(kurt_hp)
    out['kurt_hp_skw'] = f_skew(kurt_hp)
    out['kurt_hp_krt'] = f_kurt(kurt_hp)
    out['kurt_hp_min'] = f_min(kurt_hp)
    out['kurt_hp_max'] = f_max(kurt_hp)
    n_cent_hp = util.hi_pass(n_cent)
    out['n_cent_hp_avg'] = f_avg(n_cent_hp)
    out['n_cent_hp_std'] = f_std(n_cent_hp)
    out['n_cent_hp_skw'] = f_skew(n_cent_hp)
    out['n_cent_hp_krt'] = f_kurt(n_cent_hp)
    out['n_cent_hp_min'] = f_min(n_cent_hp)
    out['n_cent_hp_max'] = f_max(n_cent_hp)
    n_spread_hp = util.hi_pass(n_spread)
    out['n_spread_hp_avg'] = f_avg(n_spread_hp)
    out['n_spread_hp_std'] = f_std(n_spread_hp)
    out['n_spread_hp_skw'] = f_skew(n_spread_hp)
    out['n_spread_hp_krt'] = f_kurt(n_spread_hp)
    out['n_spread_hp_min'] = f_min(n_spread_hp)
    out['n_spread_hp_max'] = f_max(n_spread_hp)
    n_skew_hp = util.hi_pass(n_skew)
    out['n_skew_hp_avg'] = f_avg(n_skew_hp)
    out['n_skew_hp_std'] = f_std(n_skew_hp)
    out['n_skew_hp_skw'] = f_skew(n_skew_hp)
    out['n_skew_hp_krt'] = f_kurt(n_skew_hp)
    out['n_skew_hp_min'] = f_min(n_skew_hp)
    out['n_skew_hp_max'] = f_max(n_skew_hp)
    n_kurt_hp = util.hi_pass(n_kurt)
    out['n_kurt_hp_avg'] = f_avg(n_kurt_hp)
    out['n_kurt_hp_std'] = f_std(n_kurt_hp)
    out['n_kurt_hp_skw'] = f_skew(n_kurt_hp)
    out['n_kurt_hp_krt'] = f_kurt(n_kurt_hp)
    out['n_kurt_hp_min'] = f_min(n_kurt_hp)
    out['n_kurt_hp_max'] = f_max(n_kurt_hp)
    flux_hp = util.hi_pass(flux)
    out['flux_hp_avg'] = f_avg(flux_hp)
    out['flux_hp_std'] = f_std(flux_hp)
    out['flux_hp_skw'] = f_skew(flux_hp)[0]
    out['flux_hp_krt'] = f_kurt(flux_hp)[0]
    out['flux_hp_min'] = f_min(flux_hp)
    out['flux_hp_max'] = f_max(flux_hp)
    rolloff_hp = util.hi_pass(rolloff)
    out['rolloff_hp_avg'] = f_avg(rolloff_hp)
    out['rolloff_hp_std'] = f_std(rolloff_hp)
    out['rolloff_hp_skw'] = f_skew(rolloff_hp)[0]
    out['rolloff_hp_krt'] = f_kurt(rolloff_hp)[0]
    out['rolloff_hp_min'] = f_min(rolloff_hp)
    out['rolloff_hp_max'] = f_max(rolloff_hp)
    slope_hp = util.hi_pass(slope)
    out['slope_hp_avg'] = f_avg(slope_hp)
    out['slope_hp_std'] = f_std(slope_hp)
    out['slope_hp_skw'] = f_skew(slope_hp)[0]
    out['slope_hp_krt'] = f_kurt(slope_hp)[0]
    out['slope_hp_min'] = f_min(slope_hp)
    out['slope_hp_max'] = f_max(slope_hp)
    flat_hp = util.hi_pass(flat)
    out['flat_hp_avg'] = f_avg(flat_hp)
    out['flat_hp_std'] = f_std(flat_hp)
    out['flat_hp_skw'] = f_skew(flat_hp)[0]
    out['flat_hp_krt'] = f_kurt(flat_hp)[0]
    out['flat_hp_min'] = f_min(flat_hp)
    out['flat_hp_max'] = f_max(flat_hp)
    return out
