"""
See:
https://github.com/adobe-research/DeepAFx/blob/main/scripts/custom_grad_example3.py
https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html
https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
"""
from typing import Any

import numpy as np
import torch


# Custom gradient example with 2 vector arguments of the function:
# f(x, y) = x^2 * np.prod(y) where x and y are vectors

# Python Class with custom analytic gradient
from torch.autograd import Variable


class TwoVecAnalytic(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(x, y)
        z = np.square(x.numpy()) * np.prod(y.numpy())
        return torch.Tensor(z)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, y, = ctx.saved_tensors
        dy = np.zeros_like(y.numpy())
        for i in range(y.shape[0]):
            dy[i] = np.square(x.numpy()).T @ grad_outputs[0].numpy() * np.prod(np.delete(y.numpy(), [i]))
        return Variable(torch.Tensor(grad_outputs[0] * 2 * x * np.prod(y.numpy()))), Variable(torch.Tensor(dy))


# Custom numerical gradient
class TwoVecNumerical(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, y, eps: float = 0.01) -> Any:
        ctx.save_for_backward(x, y)
        ctx.eps = eps
        z = torch.square(x) * torch.prod(y)
        return z

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, y, = ctx.saved_tensors
        J_plus = TwoVecNumerical.apply(x + ctx.eps, y)
        J_minus = TwoVecNumerical.apply(x - ctx.eps, y)
        gradx = (J_plus - J_minus) / (2 * ctx.eps)

        dy = torch.zeros_like(y)
        for i in range(y.shape[0]):
            y[i] += ctx.eps
            J_plus = TwoVecNumerical.apply(x, y)
            y[i] -= 2 * ctx.eps
            J_minus = TwoVecNumerical.apply(x, y)
            grady = (J_plus - J_minus) / (2 * ctx.eps)
            y[i] += ctx.eps
            dy[i] = grady.T @ grad_outputs[0]
        return gradx * grad_outputs[0], dy


x = torch.randn(3, 1, requires_grad=True)
y = torch.randn(3, 1, requires_grad=True)
print("Input tensors:\n", x, y)
z_autograd = torch.square(x) * torch.prod(y)
z_autograd.backward(torch.ones_like(z_autograd))
print("Autograd wrt x:\n", x.grad)
print("Autograd wrt y: ", y.grad)
x.grad = None
y.grad = None
z_analytic = TwoVecAnalytic.apply(x, y)
z_analytic.backward(torch.ones_like(z_analytic))
print("Analytic wrt x:\n", x.grad)
print("Analytic wrt y: ", y.grad)
x.grad = None
y.grad = None
z_numeric = TwoVecNumerical.apply(x, y)
z_numeric.backward(torch.ones_like(z_numeric))
print("Numeric wrt x:\n", x.grad)
print("Numeric wrt y: ", y.grad)
