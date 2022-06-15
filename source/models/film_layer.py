import torch
from torch import nn


class FilmLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(FilmLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(self.input_size, 2 * self.output_size)      # for bias and slope

    def forward(self, x):
        out = self.linear(x)
        out = torch.chunk(out, 2, -1)
        return out
