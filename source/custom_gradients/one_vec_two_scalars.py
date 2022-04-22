"""
See:
https://github.com/adobe-research/DeepAFx/blob/main/scripts/custom_grad_example3.py
https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html
https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
"""
from typing import Any

import numpy as np
import torch


# Custom gradient example with one vector and one scalr arguments of the function:
# f(x, y) = x^2 * y where x is a vector

# Python Class with custom analytic gradient
class OneVecTwoScalAnalytic(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(x, y)
        z = torch.square(x) * y
        return z

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, y, = ctx.saved_tensors
        return grad_outputs[0] * 2 * x * y, grad_outputs[0] * torch.square(x)


# Custom numerical gradient
class OneVecTwoScalNumerical(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, y, eps: float = 0.01) -> Any:
        ctx.save_for_backward(x, y)
        ctx.eps = eps
        z = torch.square(x) * y
        return z

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, y, = ctx.saved_tensors
        J_plus = OneVecTwoScalNumerical.apply(x + ctx.eps, y)
        J_minus = OneVecTwoScalNumerical.apply(x - ctx.eps, y)
        gradx = (J_plus - J_minus) / (2 * ctx.eps)
        J_plus = OneVecTwoScalNumerical.apply(x, y + ctx.eps)
        J_minus = OneVecTwoScalNumerical.apply(x, y - ctx.eps)
        grady = (J_plus - J_minus) / (2 * ctx.eps)
        return gradx * grad_outputs[0], grady * grad_outputs[0]


x = torch.randn(3, 1, requires_grad=True)
y = torch.randn(1, requires_grad=True)
print("Input tensors:\n", x, y)
z_autograd = torch.square(x) * y
z_autograd.backward(torch.ones_like(z_autograd))
print("Autograd wrt x:\n", x.grad)
print("Autograd wrt y: ", y.grad)
x.grad = None
y.grad = None
z_analytic = OneVecTwoScalAnalytic.apply(x, y)
z_analytic.backward(torch.ones_like(z_analytic))
print("Analytic wrt x:\n", x.grad)
print("Analytic wrt y: ", y.grad)
x.grad = None
y.grad = None
z_numeric = OneVecTwoScalNumerical.apply(x, y)
z_numeric.backward(torch.ones_like(z_numeric))
print("Numeric wrt x:\n", x.grad)
print("Numeric wrt y: ", y.grad)
