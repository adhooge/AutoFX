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
    assert (cent == 0.25).all


def test_spectral_centroid_shape():
    mag = np.random.random(10)
    cent = Ft.spectral_centroid(mag=mag)
    assert cent.shape == (1,)


def test_spectral_centroid_value(mag_synthetic_np):
    cent = Ft.spectral_centroid(mag=mag_synthetic_np)
    assert cent == 0.25
