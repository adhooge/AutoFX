from typing import Any

import pandas as pd
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from ignite.metrics import Accuracy
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.metrics.confusion_matrix import ConfusionMatrix
import torchaudio


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
                 max_iter: int, learning_rate: float = 0.0001,
                 tol: float = 1e-4, n_iter_no_change: int = 10):
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
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change

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
        self.log("train_loss", loss)
        self.logger.experiment.add_scalar("Cross-entropy loss", loss, global_step=self.global_step)
        classes = MLPClassifier._to_one_hot(pred, self.output_size)
        self.prec.reset()
        self.accuracy.reset()
        self.recall.reset()
        self.confusion_matrix.reset()
        self.prec.update((classes, label))
        self.accuracy.update((classes, label))
        self.recall.update((classes, label))
        # self.confusion_matrix.update((classes, label))
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
        # confusion_matrix = self.confusion_matrix.compute()
        # fig = util.make_confusion_matrix(confusion_matrix.numpy(),
        #                                 group_names=CLASSES)
        # self.logger.experiment.add_figure("Metrics/Confusion_matrix",
        #                                  fig, global_step=self.global_step)
        return loss

    def configure_optimizers(self):
        if self.solver == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError
        return optimizer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(8192, hop_length=512)

    def forward(self, audio, rate):
        feat = []
        stft = self.spectrogram(audio)
