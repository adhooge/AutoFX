from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SimpleMLP(pl.LightningModule):
    """A simple MLP trained directly on features."""
    def __init__(self, num_features, hidden_size, num_params):
        super(SimpleMLP, self).__init__()
        self.fcl1 = nn.Linear(num_features, hidden_size)
        self.fcl2 = nn.Linear(hidden_size, num_params)
        self.batchnorm = nn.BatchNorm1d(hidden_size, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.loss = nn.MSELoss()

    def forward(self, x, *args, **kwargs) -> Any:
        out = self.fcl1(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.fcl2(out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        feat, label = batch
        pred = self.forward(feat)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("MSELoss/Train", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Train", scalars, global_step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        feat, label = batch
        pred = self.forward(feat)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("MSELoss/Validation", loss, global_step=self.global_step)
        scalars = {}
        for (i, val) in enumerate(torch.mean(torch.abs(pred - label), 0)):
            scalars[f'{i}'] = val
        self.logger.experiment.add_scalars("Param_distance/Validation", scalars, global_step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01)
        return optimizer


class SimpleDataset(Dataset):
    def __init__(self, data, target):
        super(SimpleDataset, self).__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.target[item]


if __name__ == "__main__":
    df = pd.read_csv("/home/alexandre/dataset/modulation_guitar_mono_cut/data.csv", index_col=0)
    df = df.drop(columns=["Unnamed: 0", "conditioning"])
    target = df.iloc[:, :5]
    feat = df.iloc[:, 5:]
    X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=0.3, random_state=2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    train_dataset = SimpleDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train.values, dtype=torch.float))
    test_dataset = SimpleDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test.values, dtype=torch.float))
    train_dl = DataLoader(train_dataset, batch_size=32, num_workers=6)
    test_dl = DataLoader(test_dataset, batch_size=32, num_workers=6)

    mlp = SimpleMLP(28, 100, 5)
    logger = TensorBoardLogger("/home/alexandre/logs/SimpleMLP", name="28juin")
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=1000, log_every_n_steps=10)

    trainer.fit(mlp, train_dl, test_dl)