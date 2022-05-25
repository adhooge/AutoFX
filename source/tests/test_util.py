import torch
import source.util as util


def test_mean_square_linreg_torch():
    y = torch.linspace(0, 10, 100)
    beta_1, beta_0 = util.mean_square_linreg_torch(y[None, :])
    assert torch.allclose(beta_1, torch.ones_like(beta_1) * 0.1010, atol=1e-5)
    assert torch.allclose(beta_0, torch.zeros_like(beta_0), atol=1e-5)


def test_mean_square_linreg_torch_batch():
    y = torch.linspace(0, 10, 100)
    y = torch.vstack([y] * 32)
    beta_1, beta_0 = util.mean_square_linreg_torch(y)
    assert torch.allclose(beta_1, torch.ones_like(beta_1) * 0.1010, atol=1e-5)
    assert torch.allclose(beta_0, torch.zeros_like(beta_0), atol=1e-5)