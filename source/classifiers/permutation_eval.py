import pandas as pd
import sklearn as skl
from sklearn.inspection import permutation_importance
import torch
import numpy as np

from source.classifiers.classifier_pytorch import MLPClassifier, ClassificationDataset

CKPT_PATH = "/home/alexandre/logs/classif1aout/guitar_Mono/version_0/checkpoints/epoch=73-step=8584.ckpt"
LOG_FILE = "/home/alexandre/logs/classif1aout/guitar_Mono/version_0/permutation_log.txt"
classif = MLPClassifier.load_from_checkpoint(CKPT_PATH,
                                             input_size=163, output_size=11, hidden_size=100,
                                             activation='sigmoid', solver='adam', max_iter=1000)

dataset = pd.read_csv('/home/alexandre/dataset/IDMT_guitar_mono_CUT_22050/out.csv', index_col=0)
subset = dataset.drop(columns=['file'])
target = subset['class']
data = subset.drop(columns=['class'])

X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(data, target, test_size=0.1, random_state=2,
                                                                        stratify=target)

sss = skl.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=1 / 5, random_state=2)
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

    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(X_train_cv)

X_test = scaler.transform(X_test)
NUM_REPEATS = 1
file = open(LOG_FILE, 'a')

dataset = ClassificationDataset(X_test, y_test)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
iterator = iter(dataloader)
precision = []
accuracy = []
recall = []
classif.prec.reset()
classif.recall.reset()
classif.accuracy.reset()
while True:
    try:
        feat, label = next(iterator)
        pred = classif(feat)
        classif.prec.update((pred, label))
        classif.recall.update((pred, label))
        classif.accuracy.update((pred, label))
    except StopIteration:
        print("Reference Precision: ", classif.prec.compute(), file=file)
        print("Reference Recall: ", classif.recall.compute(), file=file)
        print("Reference Accuracy: ", classif.accuracy.compute(), file=file)
        break

for n in range(NUM_REPEATS):
    for c in range(X_test.shape[1]):
        shuffled = X_test.copy()
        shuffled[:, c] = np.random.permutation(shuffled[:, c])
        # print(shuffled)
        # print(y_test)
        dataset = ClassificationDataset(shuffled, y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        print("Shuffling column ", c, file=file)
        iterator = iter(dataloader)
        precision = []
        accuracy = []
        recall = []
        classif.prec.reset()
        classif.recall.reset()
        classif.accuracy.reset()
        while True:
            try:
                feat, label = next(iterator)
                pred = classif(feat)
                classif.prec.update((pred, label))
                classif.recall.update((pred, label))
                classif.accuracy.update((pred, label))
            except StopIteration:
                print("Precision: ", classif.prec.compute(), file=file)
                print("Recall: ", classif.recall.compute(), file=file)
                print("Accuracy: ", classif.accuracy.compute(), file=file)
                break
