"""
Implementation based on Martinez Ramirez et al., 2021
"""

import torch
import torch.nn.functional as F
import auraloss


def compute_time_delay(original, pred):
    original_fft = torch.fft.fft(original)
    pred_fft = torch.fft.fft(pred)
    xcorr_fft = torch.multiply(torch.conj(original_fft), pred_fft)
    xcorr = torch.fft.ifft(xcorr_fft)
    time_shift = torch.argmax(torch.abs(xcorr), dim=-1)
    return time_shift


def time_align_signals(original, pred):
    time_shift = compute_time_delay(original, pred)
    original_shifted = original[:, :-time_shift]
    pred_shifted = pred[:, time_shift:]
    return original_shifted, pred_shifted


def custom_time_loss(output, target):
    target_aligned, output_aligned = time_align_signals(target, output)
    loss_plus = F.l1_loss(target_aligned, output_aligned)
    loss_minus = F.l1_loss(target_aligned, -1 * output_aligned)
    return torch.min(loss_minus, loss_plus)


def custom_spectral_loss(output, target, fft_size: int = 1024):
    target_aligned, output_aligned = time_align_signals(target, output)
    target_fft = torch.fft.fft(target_aligned, fft_size=fft_size, dim=-1)
    output_fft = torch.fft.fft(output_aligned, fft_size=fft_size, dim=-1)
    target_mag = torch.abs(target_fft)
    output_mag = torch.abs(output_fft)
    loss = F.mse_loss(target_mag, output_mag)
    log_loss = F.mse_loss(torch.log(target_mag), torch.log(output_mag))
    return loss + log_loss
