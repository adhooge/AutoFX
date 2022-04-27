"""
Implementation based on Martinez Ramirez et al., 2021
"""

import torch
import torch.nn.functional as F


def compute_time_delay(original, pred):
    original_fft = torch.fft.fft(original)
    pred_fft = torch.fft.fft(pred)
    xcorr_fft = torch.multiply(torch.conj(original_fft), pred_fft)
    xcorr = torch.fft.ifft(xcorr_fft)
    time_shift = torch.argmax(torch.abs(xcorr), dim=-1)
    return time_shift
