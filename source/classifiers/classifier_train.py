import pandas as pd
import source.util as util
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import source.classifiers.classifier_pytorch as torch_clf

CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb',
           'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']

dataset = pd.read_csv('/home/alexandre/dataset/IDMT_FULL_CUT_22050/out.csv', index_col=0)
subset = dataset.drop(columns=['file'])
target = subset['class']
data = subset.drop(columns=['class'])
print(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=2,
                                                    stratify=target)

sss = StratifiedShuffleSplit(n_splits=9, test_size=1 / 9, random_state=2)
i = 0
for train_index, valid_index in sss.split(X_train, y_train):
    print("Working on fold", i)
    i += 1
    X_train_cv = X_train.iloc[train_index]
    X_valid_cv = X_train.iloc[valid_index]
    y_train_cv = y_train.iloc[train_index]
    y_valid_cv = y_train.iloc[valid_index]
    X_train_cv = torch.tensor(X_train_cv.values, dtype=torch.float)
    X_valid_cv = torch.tensor(X_valid_cv.values, dtype=torch.float)
    y_train_cv = torch.tensor(y_train_cv.values)
    y_valid_cv = torch.tensor(y_valid_cv.values)

    scaler = torch_clf.TorchStandardScaler()
    scaler.fit(X_train_cv)
    X_train_scaled = scaler.transform(X_train_cv.clone())
    X_valid_scaled = scaler.transform(X_valid_cv.clone())

    train_dataset = torch_clf.ClassificationDataset(X_train_scaled, y_train_cv)
    valid_dataset = torch_clf.ClassificationDataset(X_valid_scaled, y_valid_cv)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                                  num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=256, num_workers=4)

    clf = torch_clf.MLPClassifier(len(data.columns), len(CLASSES), 100, activation='sigmoid', solver='adam',
                                  max_iter=200, learning_rate=0.002)

    logger = TensorBoardLogger("/home/alexandre/logs/classif21juin", name="9fold_cross-validation")
    # early_stop_callback = EarlyStopping(monitor="train_loss",
    #                                     min_delta=clf.tol,
    #                                    patience=clf.n_iter_no_change)
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=clf.max_iter,
                         accelerator='ddp',
                         auto_select_gpus=True, log_every_n_steps=10,
                         )  # callbacks=[early_stop_callback])

    trainer.fit(clf, train_dataloader, valid_dataloader)
