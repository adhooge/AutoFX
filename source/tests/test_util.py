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


def test_approx_argmax_shape():
    arr = torch.eye(10)
    estim = util.approx_argmax(arr)
    assert estim.shape == (10, 1)


def test_approx_argmax():
    arr = torch.eye(10)
    estim = util.approx_argmax(arr, beta=1000)
    ground_truth = torch.arange(10, dtype=estim.dtype)
    assert torch.allclose(estim, ground_truth[:, None], atol=0.1)


def test_approx_argmax2_shape():
    arr = torch.eye(10)
    estim = util.approx_argmax2(arr)
    assert estim.shape == (10, 1)


def test_approx_argmax2():
    arr = torch.eye(10)
    estim = util.approx_argmax2(arr)
    ground_truth = torch.arange(10, dtype=estim.dtype)
    assert torch.allclose(estim, ground_truth[:, None], atol=0.1)