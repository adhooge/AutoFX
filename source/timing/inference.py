import timeit


setup = """
import torch
from source.models.AutoFX import AutoFX
CHECKPOINT = "/home/alexandre/logs/dmd_12aoutFeat/lightning_logs/version_1/checkpoints/epoch=19-step=27920.ckpt"
model = AutoFX.load_from_checkpoint(CHECKPOINT)
model.freeze()
audio = torch.randn((1, 1, 35000))
feat = torch.randn((1, 48))
cond = torch.zeros((1, 6))
"""

code = """
model.forward(audio, feat, cond)
"""

print(timeit.timeit(setup=setup,
                    stmt=code,
                    number=1000))