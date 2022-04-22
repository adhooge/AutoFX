"""
See:
https://github.com/adobe-research/DeepAFx/blob/main/scripts/custom_grad_example1.py
https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html
https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
"""
from typing import Any

import numpy as np
import torch


# Custom gradient example with one scalar argument of the function:
# f(x, y) = x^2 + y

# Python Class with custom analytic gradient
class Two_scalars_analytic(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, y, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(x, y)
        z = torch.square(x) + y
        return z

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, y,  = ctx.saved_tensors
        return grad_outputs[0] * 2 * x, grad_outputs[0]


# Custom numerical gradient
class Two_scalars_numerical_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, y,eps: float = 0.01) -> Any:
        ctx.save_for_backward(x, y)
        ctx.eps = eps
        z = torch.square(x) + y
        return z

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, y, = ctx.saved_tensors
        J_plus = Two_scalars_numerical_gradient.apply(x + ctx.eps, y)
        J_minus = Two_scalars_numerical_gradient.apply(x - ctx.eps, y)
        gradx = (J_plus - J_minus)/(2*ctx.eps)
        J_plus = Two_scalars_numerical_gradient.apply(x, y + ctx.eps)
        J_minus = Two_scalars_numerical_gradient.apply(x, y - ctx.eps)
        grady = (J_plus - J_minus)/(2*ctx.eps)
        return gradx * grad_outputs[0], grady * grad_outputs[0]


x = torch.randn(1, requires_grad=True)
y = torch.randn(1, requires_grad=True)
print("Input tensors: ", x, y)
z_autograd = torch.square(x) + y
z_autograd.backward()
print("Autograd wrt x: ", x.grad)
print("Autograd wrt y: ", y.grad)
x.grad = None
y.grad = None
z_analytic = Two_scalars_analytic.apply(x, y)
z_analytic.backward()
print("Analytic wrt x: ", x.grad)
print("Analytic wrt y: ", y.grad)
x.grad = None
y.grad = None
z_numeric = Two_scalars_numerical_gradient.apply(x, y)
z_numeric.backward()
print("Numeric wrt x: ", x.grad)
print("Numeric wrt y: ", y.grad)