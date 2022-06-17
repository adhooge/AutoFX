import timeit
import torch
import torchaudio

SOME_SOUND = "/home/soubaboy/Téléchargements/distortion.wav"
MODEL_PATH = "/home/soubaboy/Téléchargements/classifier_v0-1.pt"

fake_input = torch.rand((1, 44100))
some_input, rate = torchaudio.load(SOME_SOUND)


p = torch.jit.load(MODEL_PATH)


print("With Fake random input: \n", timeit.timeit('p.forward(fake_input)', 'from __main__ import p, fake_input', number=10))
print("With actual input: \n", timeit.timeit('p.forward(some_input)', 'from __main__ import p, some_input', number=10))

