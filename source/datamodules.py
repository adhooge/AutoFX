import pathlib

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader
from cfgv import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from datasets import FeatureInDomainDataset, FeatureOutDomainDataset


class FeaturesDataModule(pl.LightningDataModule):
    def __init__(self, clean_dir: pathlib.Path or str, processed_dir: pathlib.Path or str,
                 out_of_domain_dir: pathlib.Path or str, batch_size: int = 32, num_workers: int = 4,
                 out_of_domain: bool = False):
        super(FeaturesDataModule, self).__init__()
        self.clean_dir = clean_dir
        self.processed_dir = processed_dir
        self.out_of_domain_dir = out_of_domain_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.out_of_domain = out_of_domain

    def setup(self, stage: Optional[str] = None) -> None:
        in_domain_full = FeatureInDomainDataset(self.processed_dir, validation=True,
                                                clean_path=self.clean_dir, processed_path=self.processed_dir)
        out_domain_full = FeatureOutDomainDataset(self.out_of_domain_dir, self.clean_dir, self.out_of_domain_dir)
        self.in_train, self.in_val = torch.utils.data.random_split(in_domain_full,
                                                                   [len(in_domain_full) - len(in_domain_full)//5, len(in_domain_full)//5])
        self.out_train, self.out_val = torch.utils.data.random_split(out_domain_full,
                                                                     [len(out_domain_full) - len(out_domain_full)//5, len(out_domain_full)//5])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        in_dataloader = DataLoader(self.in_train, self.batch_size, num_workers=self.num_workers,
                                   shuffle=True)
        out_dataloader = DataLoader(self.out_train, self.batch_size, num_workers=self.num_workers,
                                    shuffle=True)
        if self.out_of_domain:
            return out_dataloader
        else:
            return in_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        in_dataloader = DataLoader(self.in_val, self.batch_size, num_workers=self.num_workers,
                                   shuffle=True)
        out_dataloader = DataLoader(self.out_val, self.batch_size, num_workers=self.num_workers,
                                    shuffle=True)
        return [in_dataloader, out_dataloader]
