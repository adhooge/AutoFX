import torch

import data.superflux as sf
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

frame_size = 2048
rate = 22050
fps = 200
bands = 24
fmin = 30
fmax = 17000
ratio = 0.5
equal = False
max_bins = 3
# from librosa
threshold = 1.1
pre_max = 0.03
post_max = 0
pre_avg = 0.1
post_avg = 0.1
combine = 30

PATH = "/home/alexandre/Music/guitar.wav"

audio, rate = torchaudio.load(PATH)
print(rate)
print(audio.shape)

filt = sf.Filter(frame_size // 2 + 1, rate, bands, fmin, fmax, equal)
filterbank = filt.filterbank

s = sf.Spectrogram(audio, rate, frame_size, fps, filterbank, log=True)
sodf = sf.SpectralODF(s, ratio, max_bins)
act = sodf.superflux()
o = sf.Onset(act, fps, online=False)
o.detect(threshold, combine, pre_avg, pre_max, post_avg, post_max)
print("Detections", o.detections)
print("Activations", o.detect_activations)

activations = o.detect_activations / torch.max(o.detect_activations)
colors = [[1, 0, 0]]*len(o.detections)
for (i, c) in enumerate(colors):
    c[0] *= activations[i]

plt.figure()
audio=audio[0]
plt.plot(np.linspace(0, len(audio)/rate, len(audio)), audio.detach().numpy())
plt.vlines(o.detections.detach().numpy(), -1, 1, colors=colors)
plt.show()
assert False
