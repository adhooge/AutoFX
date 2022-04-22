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
# f(x) = x^2

# Python function with no custom gradient
def foo_no_grad(x):
    y = np.square(x)
    return torch.Tensor(y)


# Python Class with custom analytic gradient
class square_custom_analytic_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, *args: Any, **kwargs: Any) -> Any:
        ctx.save_for_backward(x)
        y = torch.square(x)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, = ctx.saved_tensors
        return grad_outputs[0] * 2 * x


# Custom numerical gradient
class square_custom_numerical_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, eps: float = 0.01) -> Any:
        ctx.save_for_backward(x)
        ctx.eps = eps
        y = torch.square(x)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x, = ctx.saved_tensors
        J_plus = square_custom_numerical_gradient.apply(x + ctx.eps)
        J_minus = square_custom_numerical_gradient.apply(x - ctx.eps)
        grad = (J_plus - J_minus)/(2*ctx.eps)
        return grad * grad_outputs[0]


x = torch.randn(1, requires_grad=True)
print("Input tensor: ", x)
y_autograd = torch.empty_like(x, requires_grad=True)
y_autograd = torch.square(x)
y_autograd.backward()
print("Autograd: ", x.grad)
x.grad = None
y_analytic = square_custom_analytic_grad.apply(x)
y_analytic.backward()
print("Analytic: ", x.grad)
x.grad = None
y_numeric = square_custom_numerical_gradient.apply(x)
y_numeric.backward()
print("Numeric: ", x.grad)
