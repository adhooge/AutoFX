from typing import Any

import torch
import torch.nn as nn
from multiband_fx import MultiBandFX


def _make_perturbation_vector(shape):
    return torch.bernoulli(torch.zeros(shape)) + 0.5


class MBFxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, cln, settings, mbfx, eps=0.001, grad_x: bool = True,
                *args: Any, **kwargs: Any) -> Any:
        ctx.eps = eps
        mbfx.set_fx_params(settings)
        ctx.mbfx = mbfx
        ctx.grad_x = grad_x
        ctx.save_for_backward(cln)
        batch_size = cln.shape[0]
        out = torch.zeros_like(cln)
        for (i, snd) in enumerate(cln):
            out[i] = mbfx(snd)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        cln, = ctx.saved_tensors
        batch_size = grad_outputs[0].shape[0]
        settings = ctx.mbfx.settings_list
        num_settings = ctx.mbfx.num_bands * len(settings[0])
        for i in range(batch_size):
            # TODO: Multiprocess implementation like DAFx?
            # Grad wrt to clean:
            snd = cln[i]
            if ctx.grad_x:
                perturbation = _make_perturbation_vector(cln.shape)
                J_plus = ctx.mbfx(snd + perturbation * ctx.eps)
                J_minus = ctx.mbfx(snd - perturbation * ctx.eps)
                gradx = (J_plus - J_minus) / (2 * ctx.eps * perturbation)
                Jx = gradx * grad_outputs[0]
            else:
                Jx = torch.ones_like(cln)
            # Grad wrt to parameters
            Jy = torch.zeros((batch_size, num_settings))
            perturbation = _make_perturbation_vector((1, num_settings))
            ctx.mbfx.add_perturbation_to_fx_params(perturbation)
            J_plus = ctx.mbfx(snd)
            ctx.mbfx.add_perturbation_to_fx_params(-2 * perturbation)
            J_minus = ctx.mbfx(snd)
            ctx.mbfx.add_perturbation_to_fx_params(perturbation)
            for j in range(num_settings):
                grady = (J_plus - J_minus) / (2 * ctx.eps * perturbation[j])
                Jy[j] = torch.dot(grad_outputs[0].T, grady)
        return Jx, Jy, None


class MBFxLayer(nn.Module):
    def __init__(self, mbfx: MultiBandFX):
        super(MBFxLayer, self).__init__()
        self.mbfx = mbfx
        self.num_params = mbfx.num_bands * len(self.mbfx.settings[0])
        self.params = nn.Parameter(torch.empty(self.num_params))
        nn.init.constant_(self.params, 0.5)
        self.mbfx.set_fx_params(self.params)

    def forward(self, x, settings=None):
        if settings is None:
            settings = self.mbfx.settings_list
        processed = MBFxFunction.apply(x, settings, self.mbfx)
        return processed  # TODO: check

    def extra_repr(self) -> str:
        return NotImplemented  # TODO
