from multiband_fx import MultiBandFX
import soundfile as sf
import sounddevice as sd
import pedalboard as pdb
import util

audio, rate = util.read_audio("../sound.wav")
mbfx = MultiBandFX(pdb.Distortion, 4)
rec = mbfx(audio, rate)

sd.play(audio.T, rate, blocking=True)
sd.play(rec.T, rate, blocking=True)