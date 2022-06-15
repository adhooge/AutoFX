import pandas as pd
import source.util as util
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=2)
X_train, X_test = torch.tensor(X_train.values, dtype=torch.float), torch.tensor(X_test.values, dtype=torch.float)
y_train, y_test = torch.tensor(y_train.values), torch.tensor(y_test.values)

scaler = torch_clf.TorchStandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train.clone())
X_test_scaled = scaler.transform(X_test.clone())

train_dataset = torch_clf.ClassificationDataset(X_train_scaled, y_train)
test_dataset = torch_clf.ClassificationDataset(X_test_scaled, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=4)


clf = torch_clf.MLPClassifier(len(data.columns), len(CLASSES), 100, activation='sigmoid', solver='adam',
                              max_iter=1000, learning_rate=0.002)

logger = TensorBoardLogger("/home/alexandre/logs/classif", name="torch")
# early_stop_callback = EarlyStopping(monitor="train_loss",
#                                     min_delta=clf.tol,
#                                    patience=clf.n_iter_no_change)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=clf.max_iter,
                     accelerator='ddp',
                     auto_select_gpus=True, log_every_n_steps=10,
                     ) #callbacks=[early_stop_callback])

trainer.fit(clf, train_dataloader, test_dataloader)
