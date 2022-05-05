from autofx_resnet import AutoFX
import pedalboard as pdb
import pathlib
from idmt_dataset import IDMTDataset
import sounddevice as sd

CHECKPOINT = "/home/alexandre/logs/resnet_chorus/version_34/checkpoints/epoch=9-step=17500.ckpt"
CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050")
dataset = IDMTDataset(PROCESSED_PATH / "fx2clean.csv", CLEAN_PATH, PROCESSED_PATH)

model = AutoFX.load_from_checkpoint(CHECKPOINT,
                                    fx=[pdb.Chorus], num_bands=1,
                                    param_range=[(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)])

model.out_of_domain = True
model.freeze()