import timeit


setup = """
import torch
from source.models.simpleMLP import SimpleMLP
CHECKPOINT = "/home/alexandre/logs/SimpleMLP11aout/211feat-conditioning-5_hidden/version_0/checkpoints/epoch=13-step=19544.ckpt"
model = SimpleMLP.load_from_checkpoint(CHECKPOINT)
model.freeze()
"""

code = """
feat = torch.randn((1, 211))
cond = torch.zeros((1, 6))
model.forward(feat, cond)
"""

print(timeit.timeit(setup=setup,
                    stmt=code,
                    number=1000))