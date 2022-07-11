import data.superflux as sf
import torchaudio

frame_size = 2048
rate = 22050
fps = 200
bands = 24
fmin = 30
fmax = 17000
ratio = 0.5
equal = False
max_bins = 3

PATH = "/home/alexandre/Music/guitar.wav"

audio, rate = torchaudio.load(PATH)

filt = sf.Filter(frame_size // 2 + 1 , rate, bands, fmin, fmax, equal)
filterbank = filt.filterbank

s = sf.Spectrogram(audio, rate, frame_size, fps, filterbank, log=True)
sodf = sf.SpectralODF(s, ratio, max_bins)
