from source.models.CAFx import CAFx
from source.models.autofx_resnet import AutoFX
import pedalboard as pdb
import pathlib
from source.data.idmt_dataset import IDMTDataset
from source.data.mbfx_dataset import MBFXDataset
import sounddevice as sd
from source.data.datasets import FeatureInDomainDataset, FeatureOutDomainDataset


PARAM_RANGE = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
CHECKPOINT = "/home/alexandre/logs/cafx/version_1/checkpoints/epoch=49-step=46800.ckpt"
# CHECKPOINT = "/home/alexandre/logs/resnet_chorus/version_50/checkpoints/epoch=39-step=35000.ckpt"
CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050_cut")
IN_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/modulation_guitar_mono_cut")
out_of_domain_dataset = FeatureOutDomainDataset(PROCESSED_PATH, CLEAN_PATH, PROCESSED_PATH,
                                                index_col=0)
in_domain_dataset = FeatureInDomainDataset(IN_DOMAIN_PATH, True, CLEAN_PATH, IN_DOMAIN_PATH)
# in_domain_dataset = MBFXDataset(IN_DOMAIN_PATH / "params.csv", CLEAN_PATH, IN_DOMAIN_PATH, 22050)
model = CAFx.load_from_checkpoint(CHECKPOINT,
                                  fx=[pdb.Chorus], num_bands=1, cond_feat=28,
                                  param_range=PARAM_RANGE)

# model = AutoFX.load_from_checkpoint(CHECKPOINT,
#                                    fx=[pdb.Chorus], num_bands=1,
#                                    param_range=PARAM_RANGE)

model.out_of_domain = True
model.freeze()
