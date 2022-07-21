import tqdm

from source.models.AutoFX import AutoFX
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
PARAM_RANGE_DELAY = [(0, 1), (0, 1), (0, 1)]
PARAM_RANGE_MODULATION = [(0.1, 10), (0, 1), (0, 20), (0, 1), (0, 1)]
CHECKPOINT = "/home/alexandre/logs/delay_and_modulation_13july/lightning_logs/version_12/checkpoints/epoch=54-step=41250.ckpt"
# CHECKPOINT = "/home/alexandre/logs/resnet_chorus/version_50/checkpoints/epoch=39-step=35000.ckpt"
CLEAN_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_dry_22050_cut")
PROCESSED_PATH = pathlib.Path("/home/alexandre/dataset/guitar_mono_delay_22050_cut")
IN_DOMAIN_PATH = pathlib.Path("/home/alexandre/dataset/modulation_delay_guitar_mono_cut")


datamodule = FeaturesDataModule(CLEAN_PATH, IN_DOMAIN_PATH, PROCESSED_PATH, seed=2,
                                conditioning=True, classes2keep=[0, 1])
datamodule.setup()
model = AutoFX.load_from_checkpoint(CHECKPOINT, num_bands=1, cond_feat=38,
                                    param_range_delay=PARAM_RANGE_DELAY,
                                    param_range_modulation=PARAM_RANGE_MODULATION)

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
    for i in tqdm.tqdm(range(32)):
        rec = model.mbfx_layer.forward(clean[i], pred[i])
        toSave[0, (2*i)*35000:(2*i + 1)*35000] = processed[i]
        toSave[0, (2 * i + 1) * 35000:2 * (i + 1) * 35000] = rec / (torch.max(torch.abs(rec)))
    sf.write("/home/alexandre/Music/delay_noFeat_ood.wav", toSave.T, 22050)