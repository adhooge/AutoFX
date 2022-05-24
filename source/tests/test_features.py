import pytest
import torch
import numpy as np
import data.features as Ft


@pytest.fixture
def mag_synthetic_torch():
    mag = torch.ones((10, 10))
    return mag


@pytest.fixture
def mag_synthetic_np():
    mag = np.ones(10)
    return mag


def test_spectral_centroid_torch_shape():
    mag = torch.rand((10, 10))
    cent = Ft.spectral_centroid(mag=mag, torch_compat=True)
    assert cent.shape == (10, 1)


def test_spectral_centroid_torch_value(mag_synthetic_torch):
    cent = Ft.spectral_centroid(mag=mag_synthetic_torch, torch_compat=True)
    assert (cent == 0.25).all()


def test_spectral_centroid_shape():
    mag = np.random.random(10)
    cent = Ft.spectral_centroid(mag=mag)
    assert cent.shape == (1,)


def test_spectral_centroid_value(mag_synthetic_np):
    cent = Ft.spectral_centroid(mag=mag_synthetic_np)
    assert cent == 0.25


def test_spectral_spread_torch_shape():
    mag = torch.rand((10, 10))
    spread = Ft.spectral_spread(mag=mag, torch_compat=True)
    assert spread.shape == (10, 1)


def test_spectral_spread_torch_value():
    mag = torch.eye(10)
    spread = Ft.spectral_spread(mag=mag, torch_compat=True)
    assert (spread == 0).all()


def test_spectral_spread_shape():
    mag = np.random.random(10)
    spread = Ft.spectral_spread(mag=mag)
    assert spread.shape == (1,)


def test_spectral_spread_value():
    mag = np.zeros(10)
    mag[-1] = 1
    spread = Ft.spectral_spread(mag=mag)
    assert spread == 0


def test_spectral_skewness_torch_shape():
    mag = torch.rand((10, 10))
    skew = Ft.spectral_skewness(mag=mag, torch_compat=True)
    assert skew.shape == (10, 1)


def test_spectral_skewness_torch_value():
    mag_sym = torch.ones((10, 10))
    skew_sym = Ft.spectral_skewness(mag=mag_sym, torch_compat=True)
    mag_left = torch.zeros((10, 10))
    mag_left[:, :4] = torch.ones((10, 4))
    mag_left[:, 0] *= 2
    skew_left = Ft.spectral_skewness(mag=mag_left, torch_compat=True)
    mag_right = torch.zeros((10, 10))
    mag_right[:, -4:] = torch.ones((10, 4))
    mag_right[:, -1] *= 2
    skew_right = Ft.spectral_skewness(mag=mag_right, torch_compat=True)
    assert torch.allclose(skew_sym, torch.zeros_like(skew_sym)) and (skew_right < 0).all() and (skew_left > 0).all()


def test_spectral_skewness_shape():
    mag = np.random.random(10)
    skew = Ft.spectral_skewness(mag=mag)
    assert skew.shape == (1,)


def test_spectral_skewness_value():
    mag_sym = np.ones(10)
    skew_sym = Ft.spectral_skewness(mag=mag_sym)
    mag_left = np.zeros(10)
    mag_left[:2] = [2, 1]
    skew_left = Ft.spectral_skewness(mag=mag_left)
    mag_right = np.zeros(10)
    mag_right[-2:] = [1, 2]
    skew_right = Ft.spectral_skewness(mag=mag_right)
    assert np.allclose(skew_sym, np.zeros_like(skew_sym))
    assert skew_right < 0
    assert skew_left > 0
