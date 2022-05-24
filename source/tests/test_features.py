import pytest
import torch
import numpy as np
import data.features as Ft
from math import sqrt

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
    mag = 1/(std*sqrt(2*torch.pi)) * torch.exp(-0.5*torch.square((x - mean)/std))
    return mag


@pytest.fixture
def gaussian_mag_np():
    x = np.linspace(0, 0.5, 100)
    std = 0.05
    mean = 0.25
    mag = 1/(std*sqrt(2*np.pi)) * np.exp(-0.5*np.square((x - mean)/std))
    return mag


def test_spectral_centroid_torch_shape():
    mag = torch.rand((10, 100))
    cent = Ft.spectral_centroid(mag=mag, torch_compat=True)
    assert cent.shape == (10, 1)


def test_spectral_centroid_torch_value(mag_synthetic_torch):
    cent = Ft.spectral_centroid(mag=mag_synthetic_torch, torch_compat=True)
    assert (cent == 0.25).all()


def test_spectral_centroid_shape():
    mag = np.random.random(100)
    cent = Ft.spectral_centroid(mag=mag)
    assert cent.shape == (1,)


def test_spectral_centroid_value(mag_synthetic_np):
    cent = Ft.spectral_centroid(mag=mag_synthetic_np)
    assert np.allclose(cent, np.ones_like(cent)*0.25)


def test_spectral_spread_torch_shape():
    mag = torch.rand((10, 100))
    spread = Ft.spectral_spread(mag=mag, torch_compat=True)
    assert spread.shape == (10, 1)


def test_spectral_spread_torch_value():
    mag = torch.eye(10)
    spread = Ft.spectral_spread(mag=mag, torch_compat=True)
    assert (spread == 0).all()


def test_spectral_spread_shape():
    mag = np.random.random(100)
    spread = Ft.spectral_spread(mag=mag)
    assert spread.shape == (1,)


def test_spectral_spread_value():
    mag = np.zeros(100)
    mag[-1] = 1
    spread = Ft.spectral_spread(mag=mag)
    assert spread == 0


def test_spectral_skewness_torch_shape():
    mag = torch.rand((10, 100))
    skew = Ft.spectral_skewness(mag=mag, torch_compat=True)
    assert skew.shape == (10, 1)


def test_spectral_skewness_torch_value():
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
    mag = np.random.random(100)
    skew = Ft.spectral_skewness(mag=mag)
    assert skew.shape == (1,)


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
    mag = torch.rand((10, 100))
    kurt = Ft.spectral_kurtosis(mag=mag, torch_compat=True)
    assert kurt.shape == (10, 1)


def test_spectral_kurtosis_torch_value(gaussian_mag_torch):
    kurt_normal = Ft.spectral_kurtosis(mag=gaussian_mag_torch, torch_compat=True)
    assert (kurt_normal == 3).all()
    mag_flat = torch.ones((10, 100))
    kurt_flat = Ft.spectral_kurtosis(mag=mag_flat, torch_compat=True)
    assert (kurt_flat < 3).all()
    mag_peak = torch.zeros((10, 100))
    mag_peak[:, 4:7] = torch.ones((10, 3))
    mag_peak[:, 5] *= 2
    kurt_peak = Ft.spectral_kurtosis(mag=mag_peak, torch_compat=True)
    assert (kurt_peak > 3).all()
