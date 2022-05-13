import pathlib

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader
from cfgv import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from datasets import FeatureInDomainDataset, FeatureOutDomainDataset


class FeaturesDataModule(pl.LightningDataModule):
    def __init__(self, clean_dir: pathlib.Path or str, processed_dir: pathlib.Path or str,
                 out_of_domain_dir: pathlib.Path or str, batch_size: int = 32, validation: bool = False):
        super(FeaturesDataModule, self).__init__()
        self.clean_dir = clean_dir
        self.processed_dir = processed_dir
        self.out_of_domain_dir = out_of_domain_dir
        self.batch_size = batch_size
        self.validation = validation

    def setup(self, stage: Optional[str] = None) -> None:
        in_domain_full = FeatureInDomainDataset(self.processed_dir, validation=self.validation,
                                                clean_path=self.clean_dir, processed_path=self.processed_dir)
        out_domain_full = FeatureOutDomainDataset(self.processed_dir, self.clean_dir, self.processed_dir)
        self.in_train, self.in_val = torch.utils.data.random_split(in_domain_full,
                                                                   [len(in_domain_full) - len(in_domain_full)//5, len(in_domain_full)//5])
        self.out_train, self.out_val = torch.utils.data.random_split(out_domain_full,
                                                                     [len(out_domain_full) - len(out_domain_full)//5, len(out_domain_full)//5])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        in_dataloader = DataLoader(self.in_train, self.batch_size)
        out_dataloader = DataLoader(self.out_train, self.batch_size)
        return [in_dataloader, out_dataloader]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        in_dataloader = DataLoader(self.in_val, self.batch_size)
        out_dataloader = DataLoader(self.out_val, self.batch_size)
        return [in_dataloader, out_dataloader]
