import timeit
import torch


setup = """
import torchaudio
from source.classifiers.classifier_pytorch import FeatureExtractor
extractor = FeatureExtractor()
audio, rate = torchaudio.load("/home/alexandre/dataset/IDMT_guitar_mono_CUT_22050/G93-76612-4411-38064.wav")
print(rate)
"""

code = """
feat = extractor(audio)
"""

print(timeit.timeit(setup=setup,
                    stmt=code,
                    number=1000))