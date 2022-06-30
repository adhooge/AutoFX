import pathlib

import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from cfgv import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from source.data.datasets import FeatureInDomainDataset, FeatureOutDomainDataset


class FeaturesDataModule(pl.LightningDataModule):
    def __init__(self, clean_dir: str, processed_dir: str,
                 out_of_domain_dir: str, batch_size: int = 32, num_workers: int = 4,
                 in_scaler_mean: list = None, in_scaler_std: list = None,
                 out_scaler_mean: list = None, out_scaler_std: list = None,
                 out_of_domain: bool = False, seed: int = None, reverb: bool = False,
                 conditioning: bool = False, class_indices: list = None, *args, **kwargs):
        super(FeaturesDataModule, self).__init__()
        self.clean_dir = clean_dir
        self.processed_dir = processed_dir
        self.out_of_domain_dir = out_of_domain_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.out_of_domain = out_of_domain
        if seed is None:
            seed = torch.randint(100000, (1, 1))
        self.seed = seed
        self.reverb = reverb
        self.in_scaler_mean = in_scaler_mean
        self.in_scaler_std = in_scaler_std
        self.out_scaler_mean = out_scaler_mean
        self.out_scaler_std = out_scaler_std
        self.conditioning = conditioning
        self.class_indices = class_indices
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        in_domain_full = FeatureInDomainDataset(self.processed_dir, validation=True,
                                                clean_path=self.clean_dir, processed_path=self.processed_dir,
                                                reverb=self.reverb, conditioning=self.conditioning)
        out_domain_full = FeatureOutDomainDataset(self.out_of_domain_dir, self.clean_dir, self.out_of_domain_dir,
                                                  index_col=0, conditioning=self.conditioning)
        self.in_train, self.in_val = torch.utils.data.random_split(in_domain_full,
                                                                   [len(in_domain_full) - len(in_domain_full)//5, len(in_domain_full)//5],
                                                                   generator=torch.Generator().manual_seed(self.seed))
        self.out_train, self.out_val = torch.utils.data.random_split(out_domain_full,
                                                                     [len(out_domain_full) - len(out_domain_full)//5, len(out_domain_full)//5],
                                                                     generator=torch.Generator().manual_seed(self.seed))
        if self.in_scaler_mean is None or self.in_scaler_std is None:
            tmp_dataloader = DataLoader(self.in_train, batch_size=len(self.in_train),
                                        num_workers=self.num_workers)
            in_domain_full.scaler.fit(next(iter(tmp_dataloader))[:][2])
        else:
            in_domain_full.scaler.mean = torch.tensor(self.in_scaler_mean)
            in_domain_full.scaler.std = torch.tensor(self.in_scaler_std)
        print("Scaler mean: ", in_domain_full.scaler.mean)
        print("Scaler std: ", in_domain_full.scaler.std)
        if self.out_scaler_std is None or self.out_scaler_mean is None:
            tmp_dataloader = DataLoader(self.out_train, batch_size=len(self.out_train),
                                        num_workers=self.num_workers)
            out_domain_full.scaler.fit(next(iter(tmp_dataloader))[:][2])
        else:
            out_domain_full.scaler.mean = torch.tensor(self.out_scaler_mean)
            out_domain_full.scaler.std = torch.tensor(self.out_scaler_std)
        print("Out Scaler mean: ", out_domain_full.scaler.mean)
        print("Out Scaler std: ", out_domain_full.scaler.std)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not self.conditioning:
            in_dataloader = DataLoader(self.in_train, self.batch_size, num_workers=self.num_workers,
                                       shuffle=True)
            out_dataloader = DataLoader(self.out_train, self.batch_size, num_workers=self.num_workers,
                                        shuffle=True)
        else:
            class_inds = [torch.nonzero(self.in_train.dataset.target_classes_subset(self.in_train.indices) == class_idx,
                                        as_tuple=True)[0]
                          for class_idx in self.class_indices]
            # class_inds = [torch.nonzero(inds in self.in_train.indices, as_tuple=True)[0] for inds in tmp]
            in_dataloader = [
                DataLoader(
                    dataset=Subset(self.in_train, inds),
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True)
                for inds in class_inds]
            class_inds = [torch.nonzero(self.out_train.dataset.target_classes_subset(self.out_train.indices) == class_idx,
                                        as_tuple=True)[0]
                          for class_idx in self.class_indices]
            # class_inds = [torch.nonzero(inds in self.in_train.indices, as_tuple=True)[0] for inds in tmp]
            out_dataloader = [
                DataLoader(
                    dataset=Subset(self.out_train, inds),
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True)
                for inds in class_inds]
        if self.out_of_domain:
            return out_dataloader
        else:
            print("PLOP", in_dataloader)
            return in_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # print("CONDITIONING: ", self.conditioning)
        in_dataloader = DataLoader(self.in_val, self.batch_size, num_workers=self.num_workers,
                                   shuffle=True)
        out_dataloader = DataLoader(self.out_val, self.batch_size, num_workers=self.num_workers,
                                    shuffle=True)
        if self.out_of_domain:
            return out_dataloader
        else:
            return in_dataloader


