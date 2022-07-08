import data.superflux as sf

frame_size = 2048
rate = 22050
bands = 24
fmin = 30
fmax = 17000
equal = False

filt = sf.Filter(frame_size // 2, rate, bands, fmin, fmax, equal)
filterbank = filt.filterbank