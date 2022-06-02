import math

import pytest
import torch
import numpy as np
import data.features as Ft
from math import sqrt


@pytest.fixture
def a440_np():
    t = np.linspace(0, 2, 32000)
    audio = np.sin(2*np.pi*440*t)
    return audio


@pytest.fixture
def a440_torch():
    t = torch.linspace(0, 4, 64000)
    audio = torch.sin(2*torch.pi*440*t + torch.pi)
    audio = torch.vstack([audio] * 10)
    return audio


@pytest.fixture
def mag_synthetic_torch():
    mag = torch.ones((10, 100))
    return mag


@pytest.fixture
def mag_synthetic_np():
    mag = np.ones(100)
    return mag


@pytest.fixture
def gaussian_mag_torch():
    x = torch.linspace(0, 0.5, 100)
    x = torch.vstack([x] * 10)
    std = 0.05
    mean = 0.25
    mag = (1 / (std * sqrt(2 * torch.pi))) * torch.exp(-0.5 * torch.square((x - mean) / std))
    return mag


@pytest.fixture
def gaussian_mag_np():
    x = np.linspace(0, 0.5, 100)
    std = 0.05
    mean = 0.25
    mag = 1 / (std * sqrt(2 * np.pi)) * np.exp(-0.5 * np.square((x - mean) / std))
    return mag


def test_spectral_centroid_torch_shape():
    mag = torch.rand((32, 513, 100))
    cent = Ft.spectral_centroid(mag=mag, torch_compat=True)
    assert cent.shape == (32, 1, 100)


def test_spectral_centroid_torch_value(mag_synthetic_torch):
    cent = Ft.spectral_centroid(mag=mag_synthetic_torch, torch_compat=True)
    assert (cent == 0.25).all()


def test_spectral_centroid_shape():
    mag = np.random.random((257, 10))
    cent = Ft.spectral_centroid(mag=mag)
    assert cent.shape == (1, 10)


def test_spectral_centroid_value(mag_synthetic_np):
    cent = Ft.spectral_centroid(mag=mag_synthetic_np)
    assert np.allclose(cent, np.ones_like(cent) * 0.25)


def test_spectral_spread_torch_shape():
    mag = torch.rand((32, 257, 100))
    spread = Ft.spectral_spread(mag=mag, torch_compat=True)
    assert spread.shape == (32, 1, 100)


def test_spectral_spread_torch_value():
    mag = torch.eye(10)
    spread = Ft.spectral_spread(mag=mag, torch_compat=True)
    assert (spread == 0).all()


def test_spectral_spread_shape():
    mag = np.random.random((257, 100))
    spread = Ft.spectral_spread(mag=mag)
    assert spread.shape == (1, 100)


def test_spectral_spread_value():
    mag = np.zeros(100)
    mag[-1] = 1
    spread = Ft.spectral_spread(mag=mag)
    assert spread == 0


def test_spectral_skewness_torch_shape():
    mag = torch.rand((32, 257, 100))
    skew = Ft.spectral_skewness(mag=mag, torch_compat=True)
    assert skew.shape == (32, 1, 100)


def test_spectral_skewness_torch_value():
    # TODO: update
    mag_sym = torch.ones((10, 100))
    skew_sym = Ft.spectral_skewness(mag=mag_sym, torch_compat=True)
    mag_left = torch.zeros((10, 100))
    mag_left[:, :4] = torch.ones((10, 4))
    mag_left[:, 0] *= 2
    skew_left = Ft.spectral_skewness(mag=mag_left, torch_compat=True)
    mag_right = torch.zeros((10, 100))
    mag_right[:, -4:] = torch.ones((10, 4))
    mag_right[:, -1] *= 2
    skew_right = Ft.spectral_skewness(mag=mag_right, torch_compat=True)
    assert torch.allclose(skew_sym, torch.zeros_like(skew_sym)) and (skew_right < 0).all() and (skew_left > 0).all()


def test_spectral_skewness_shape():
    mag = np.random.random((257, 10))
    skew = Ft.spectral_skewness(mag=mag)
    assert skew.shape == (1, 10)


def test_spectral_skewness_value():
    mag_sym = np.ones(100)
    skew_sym = Ft.spectral_skewness(mag=mag_sym)
    mag_left = np.zeros(100)
    mag_left[:2] = [2, 1]
    skew_left = Ft.spectral_skewness(mag=mag_left)
    mag_right = np.zeros(100)
    mag_right[-2:] = [1, 2]
    skew_right = Ft.spectral_skewness(mag=mag_right)
    assert np.allclose(skew_sym, np.zeros_like(skew_sym))
    assert skew_right < 0
    assert skew_left > 0


def test_spectral_kurtosis_torch_shape():
    mag = torch.rand((32, 257, 100))
    kurt = Ft.spectral_kurtosis(mag=mag, torch_compat=True)
    assert kurt.shape == (32, 1, 100)


def test_spectral_kurtosis_torch_value(gaussian_mag_torch):
    kurt_normal = Ft.spectral_kurtosis(mag=gaussian_mag_torch, torch_compat=True)
    spread_normal = Ft.spectral_spread(mag=gaussian_mag_torch, torch_compat=True)
    kurt_normal = kurt_normal / torch.pow(spread_normal, 2)
    assert torch.allclose(kurt_normal, torch.ones_like(kurt_normal) * 3, atol=1e-4, rtol=1e-4)
    mag_flat = torch.ones((10, 100))
    kurt_flat = Ft.spectral_kurtosis(mag=mag_flat, torch_compat=True)
    spread_flat = Ft.spectral_spread(mag=mag_flat, torch_compat=True)
    kurt_flat = kurt_flat / torch.pow(spread_flat, 2)
    assert (kurt_flat < 3).all()
    mag_peak = torch.zeros((10, 100))
    mag_peak[:, 40:50] = torch.pow(torch.vstack([torch.linspace(0, 1, 10)] * 10), 5)
    mag_peak[:, 50:60] = torch.pow(torch.vstack([torch.linspace(1, 0, 10)] * 10), 5)
    kurt_peak = Ft.spectral_kurtosis(mag=mag_peak, torch_compat=True)
    spread_peak = Ft.spectral_spread(mag=mag_peak, torch_compat=True)
    kurt_peak = kurt_peak / torch.pow(spread_peak, 2)
    assert (kurt_peak > 3).all()


def test_spectral_kurtosis_shape():
    mag = np.random.random((129, 10))
    kurt = Ft.spectral_kurtosis(mag=mag)
    assert kurt.shape == (1, 10)


def test_spectral_kurtosis_value(gaussian_mag_np):
    kurt_normal = Ft.spectral_kurtosis(mag=gaussian_mag_np)
    spread_normal = Ft.spectral_spread(mag=gaussian_mag_np)
    kurt_normal = kurt_normal / np.power(spread_normal, 2)
    assert np.allclose(kurt_normal, np.ones_like(kurt_normal)*3, atol=1e-4, rtol=1e-4)
    mag_flat = np.ones(100)
    kurt_flat = Ft.spectral_kurtosis(mag=mag_flat)
    spread_flat = Ft.spectral_spread(mag=mag_flat)
    kurt_flat = kurt_flat / np.power(spread_flat, 2)
    assert (kurt_flat < 3)
    mag_peak = np.zeros(100)
    mag_peak[40:50] = np.power(np.linspace(0, 1, 10), 5)
    mag_peak[50:60] = np.power(np.linspace(1, 0, 10), 5)
    kurt_peak = Ft.spectral_kurtosis(mag=mag_peak)
    spread_peak = Ft.spectral_spread(mag=mag_peak)
    kurt_peak = kurt_peak / np.power(spread_peak, 2)
    assert (kurt_peak > 3)


def test_flux_torch_shape():
    signal = torch.rand((32, 256, 10))
    flux = Ft.spectral_flux(signal, torch_compat=True)
    assert flux.shape == (32, 1, 10)


def test_flux_np_shape():
    signal = np.random.random((256, 10))
    flux = Ft.spectral_flux(signal)
    assert flux.shape == (1, 10)


def test_pitch_curve(a440_np):
    f0 = Ft.pitch_curve(a440_np, 16000, 80, 1000)
    assert np.allclose(f0, np.ones_like(f0) * 440, atol=0.1, rtol=0.01)

# TODO: Find a way to test torch version?


def test_phase_fmax(a440_np):
    linregerr = Ft.phase_fmax(a440_np)
    print(linregerr)
    assert np.allclose(linregerr, np.zeros_like(linregerr), atol=0.5)


def test_phase_fmax_torch(a440_torch):
    linregerr = Ft.phase_fmax_batch(a440_torch)
    assert torch.allclose(linregerr, torch.zeros_like(linregerr), atol=1)


def test_roll_off_torch():
    # (10, 5, 100)
    signal = torch.linspace(1, 100, 100)
    signal = torch.hstack([signal] * 5)
    signals = torch.vstack([signal] * 10)
    rolloff = Ft.spectral_rolloff(signals.view(10, 5, 100), torch_compat=True)
    assert torch.allclose(rolloff, torch.ones_like(rolloff) * 98)


def test_slope_torch():
    # (10, 5, 100)
    signal = torch.linspace(100, 1, 100)
    signal = torch.stack([signal]*5, dim=0)
    signals = torch.stack([signal] * 10, dim=0)
    slopes = Ft.spectral_slope(signals, torch_compat=True)
    assert torch.allclose(slopes, -torch.ones_like(slopes))


def test_flatness_torch():
    signal = torch.linspace(1, 10, 10)
    signal = torch.stack([signal] * 5, dim=0)
    signals = torch.stack([signal] * 10, dim=0)
    flatness = Ft.spectral_flatness(mag=signals, torch_compat=True)
    ground_truth = math.factorial(10)**0.1 / 5.5
    assert torch.allclose(flatness, torch.ones_like(flatness)*ground_truth, atol=1e-2)