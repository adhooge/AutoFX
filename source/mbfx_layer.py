from typing import Any

import torch
import torch.nn as nn
from multiband_fx import MultiBandFX
from torch.autograd import gradcheck
import pedalboard as pdb


def _make_perturbation_vector(shape):
    return torch.bernoulli(torch.zeros(shape)) + 0.5


class MBFxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, cln, settings, mbfx, rate, param_range, fake_num_bands: int = None,
                eps=0.001, grad_x: bool = True, *args: Any, **kwargs: Any) -> Any:
        if fake_num_bands is None:
            fake_num_bands = mbfx.num_bands
        ctx.fake_num_bands = fake_num_bands
        ctx.eps = eps
        ctx.mbfx = mbfx
        ctx.grad_x = grad_x
        ctx.rate = rate
        ctx.param_range = param_range
        ctx.save_for_backward(cln, settings)
        tmp = cln.clone()
        if tmp.ndim == 1:
            tmp = tmp[None, :]
        out = torch.zeros_like(tmp)
        for (i, snd) in enumerate(tmp):
            mbfx.set_fx_params(settings, flat=True, param_range=param_range)
            out[i] = mbfx(snd, rate)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        cln, settings, = ctx.saved_tensors
        if cln.ndim == 1:
            cln = cln[None, :]
        batch_size = grad_outputs[0].shape[0]
        mbfx = ctx.mbfx
        num_settings = ctx.mbfx.total_num_params_per_band * ctx.fake_num_bands
        for i in range(batch_size):
            # TODO: Multiprocess implementation like DAFx?
            # Grad wrt to clean:
            mbfx.set_fx_params(settings, flat=True)
            snd = cln[i]
            if ctx.grad_x:
                perturbation = _make_perturbation_vector(cln.shape)
                J_plus = mbfx(snd + perturbation * ctx.eps, ctx.rate)
                J_minus = mbfx(snd - perturbation * ctx.eps, ctx.rate)
                gradx = (J_plus - J_minus) / (2 * ctx.eps * perturbation)
                Jx = gradx * grad_outputs[0]
            else:
                Jx = torch.ones_like(cln)
            # Grad wrt to parameters
            Jy = torch.zeros((batch_size, num_settings))
            perturbation = _make_perturbation_vector((num_settings))
            mbfx.fake_add_perturbation_to_fx_params(perturbation * ctx.eps, ctx.param_range, ctx.fake_num_bands)
            J_plus = mbfx(snd, ctx.rate)
            mbfx.fake_add_perturbation_to_fx_params(-2 * perturbation * ctx.eps, ctx.param_range, ctx.fake_num_bands)
            J_minus = mbfx(snd, ctx.rate)
            mbfx.fake_add_perturbation_to_fx_params(perturbation * ctx.eps, ctx.param_range, ctx.fake_num_bands)
            for j in range(num_settings):
                grady = (J_plus - J_minus) / (2 * ctx.eps * perturbation[j])
                Jy[i][j] = grad_outputs[0] @ grady.T
        return Jx, Jy, None, None, None, None


class MBFxLayer(nn.Module):
    def __init__(self, mbfx: MultiBandFX, rate, param_range, fake_num_bands: int = None):
        super(MBFxLayer, self).__init__()
        if fake_num_bands is None:
            fake_num_bands = mbfx.num_bands
        self.fake_num_bands = fake_num_bands
        self.mbfx = mbfx
        self.num_params = fake_num_bands * self.mbfx.total_num_params_per_band
        self.params = nn.Parameter(torch.empty(self.num_params))
        nn.init.constant_(self.params, 0.5)
        self.param_range = param_range
        self.mbfx.set_fx_params(self.params, flat=True, param_range=self.param_range)
        self.rate = rate

    def forward(self, x, settings=None):
        if settings is None:
            settings = self.params
        processed = MBFxFunction.apply(x, settings, self.mbfx, self.rate, self.param_range, self.fake_num_bands)
        return processed  # TODO: check

    def extra_repr(self) -> str:
        return NotImplemented  # TODO


if __name__ == '__main__':
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    function = MBFxFunction.apply
    mbfx = MultiBandFX([pdb.Distortion, pdb.Gain], 4)
    settings = torch.tensor([0.5] * 8)
    inputs = (torch.randn(4, 22050, requires_grad=True),
              settings, mbfx, 22050, [(0, 10), (0, 10)])
    test = gradcheck(function, inputs, eps=1e-6, atol=1e-4)
    print(test)
