from source.models.CAFx import CAFx
from source.models.autofx_resnet import AutoFX
import pedalboard as pdb
import pathlib
from source.data.idmt_dataset import IDMTDataset
from source.data.mbfx_dataset import MBFXDataset
import sounddevice as sd
from source.data.datasets import FeatureInDomainDataset, FeatureOutDomainDataset
from source.data.datamodules import FeaturesDataModule
import torch
import soundfile as sf

WRITE_AUDIO = True
PARAM_RANGE = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
CHECKPOINT = "/home/alexandre/logs/finetuning29juin/lightning_logs/version_12/checkpoints/epoch=114-step=48210.ckpt"
# CHECKPOINT = "/home/alexandre/logs/resnet_chorus/version_50/checkpoints/epoch=39-step=35000.ckpt"
CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_modulation_22050_cut")
IN_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/modulation_guitar_mono_cut")
out_of_domain_dataset = FeatureOutDomainDataset(PROCESSED_PATH, CLEAN_PATH, PROCESSED_PATH,
                                                index_col=0)
in_domain_dataset = FeatureInDomainDataset(IN_DOMAIN_PATH, True, CLEAN_PATH, IN_DOMAIN_PATH)
# in_domain_dataset = MBFXDataset(IN_DOMAIN_PATH / "params.csv", CLEAN_PATH, IN_DOMAIN_PATH, 22050)

datamodule = FeaturesDataModule(CLEAN_PATH, IN_DOMAIN_PATH, PROCESSED_PATH, seed=2)
datamodule.setup()
model = CAFx.load_from_checkpoint(CHECKPOINT,
                                  fx=[pdb.Chorus], num_bands=1, cond_feat=28,
                                  param_range=PARAM_RANGE)

# model = AutoFX.load_from_checkpoint(CHECKPOINT,
#                                    fx=[pdb.Chorus], num_bands=1,
#                                    param_range=PARAM_RANGE)

model.out_of_domain = True
model.freeze()

if WRITE_AUDIO:
    datamodule.out_of_domain = True
    batch = next(iter(datamodule.val_dataloader()))
    clean, processed, feat = batch
    pred = model.forward(processed, feat)
    # print(torch.square(pred - label))
    toSave = torch.zeros((1, 64 * 35000))
    for i in range(32):
        rec = model.mbfx_layer.forward(clean[i], pred[i])
        toSave[0, (2*i)*35000:(2*i + 1)*35000] = processed[i]
        toSave[0, (2 * i + 1) * 35000:2 * (i + 1) * 35000] = rec / (torch.max(torch.abs(rec)))
    sf.write("/home/alexandre/Music/finetuned29juin_12.wav", toSave.T, 22050)