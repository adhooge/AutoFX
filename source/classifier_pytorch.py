from typing import Any

import pandas as pd
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from ignite.metrics import Accuracy
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.metrics.confusion_matrix import ConfusionMatrix

from pytorch_lightning.loggers import TensorBoardLogger

import source.util as util


class TorchStandardScaler:
    """
    from    https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/8
    """
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


class ClassificationDataset(Dataset):
    """
    Simple Dataset for iterating through samples to classify
    """

    def __init__(self, features, labels):
        super(ClassificationDataset, self).__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


class MLPClassifier(pl.LightningModule):
    @staticmethod
    def _activation(activation):
        if activation == 'logistic' or activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise NotImplementedError

    @staticmethod
    def _to_one_hot(batch, num_classes):
        idx = torch.argmax(batch, dim=-1, keepdim=False)
        out = F.one_hot(idx, num_classes=num_classes)
        return out

    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int, activation: str, solver: str,
                 max_iter: int, learning_rate: float = 0.001):
        super(MLPClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = MLPClassifier._activation(activation)
        self.solver = solver
        self.max_iter = max_iter
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.prec = Precision()
        self.recall = Recall()
        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=output_size)
        self.learning_rate = learning_rate

    def forward(self, x, *args, **kwargs) -> Any:
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        feat, label = batch
        pred = self.forward(feat)
        loss = self.loss(pred, label)
        self.logger.experiment.add_scalar("Cross-entropy loss", loss, global_step=self.global_step)
        classes = MLPClassifier._to_one_hot(pred, self.output_size)
        self.prec.reset()
        self.accuracy.reset()
        self.recall.reset()
        self.confusion_matrix.reset()
        self.prec.update((classes, label))
        self.accuracy.update((classes, label))
        self.recall.update((classes, label))
        self.confusion_matrix.update((classes, label))
        precision = self.prec.compute()
        self.logger.experiment.add_scalars("Metrics/Precision",
                                           dict(zip(CLASSES, precision)),
                                           global_step=self.global_step)
        accuracy = self.accuracy.compute()
        self.logger.experiment.add_scalar("Metrics/Accuracy",
                                           accuracy,
                                           global_step=self.global_step)
        recall = self.recall.compute()
        self.logger.experiment.add_scalars("Metrics/Recall",
                                           dict(zip(CLASSES, recall)),
                                           global_step=self.global_step)
        confusion_matrix = self.confusion_matrix.compute()
        fig = util.make_confusion_matrix(confusion_matrix.numpy(),
                                         group_names=CLASSES)
        self.logger.experiment.add_figure("Metrics/Confusion_matrix",
                                          fig, global_step=self.global_step)
        return loss

    def configure_optimizers(self):
        if self.solver == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError
        return optimizer


CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb',
           'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']

dataset = pd.read_csv('/home/alexandre/dataset/full_dataset.csv', index_col=0)
subset = dataset.drop(columns=['flux_min'])
target = []
for fx in subset['target_name']:
    target.append(util.idmt_fx2class_number(fx))
data = subset.drop(columns=['target_name'])
print(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=2)
X_train, X_test = torch.tensor(X_train.values, dtype=torch.float), torch.tensor(X_test.values, dtype=torch.float)
y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)

scaler = TorchStandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train.clone())
X_test_scaled = scaler.transform(X_test.clone())

train_dataset = ClassificationDataset(X_train_scaled, y_train)
test_dataset = ClassificationDataset(X_test_scaled, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=4)


clf = MLPClassifier(len(data.columns), len(CLASSES), 100, activation='sigmoid', solver='adam',
                    max_iter=500)

logger = TensorBoardLogger("/home/alexandre/logs", name="classifier")
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=clf.max_iter,
                     accelerator='ddp',
                     auto_select_gpus=True, log_every_n_steps=10)

trainer.fit(clf, train_dataloader, test_dataloader)
