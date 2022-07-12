import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from source.classifiers.classifier_pytorch import MLPClassifier

import source.classifiers.classifier_pytorch as torch_clf

CHECKPOINT = "/home/alexandre/logs/classif4july/guitar_Mono/version_1/checkpoints/epoch=66-step=20770.ckpt"

clf = MLPClassifier.load_from_checkpoint(CHECKPOINT, input_size=163,
                                         output_size=11, hidden_size=100,
                                         activation='sigmoid', solver='adam',
                                         max_iter=1000)


class ClassifierPipeline(nn.Module):
    def __init__(self, classifier, feat_extractor, scaler, remover, with_argmax: bool = True):
        super(ClassifierPipeline, self).__init__()
        self.clf = classifier
        # self.feat_extractor = torch.jit.trace(feat_extractor, example)
        self.feat_extractor = feat_extractor
        self.scaler = scaler
        self.remover = remover
        self.with_argmax = with_argmax
        self.pipeline = nn.Sequential(
            remover,
            feat_extractor,
            scaler,
            clf
        )

    def forward(self, audio):
        cut_audio = self.remover(audio)
        feat = self.feat_extractor(cut_audio)
        scaled_feat = self.scaler(feat)
        pred = self.clf(scaled_feat)
        if self.with_argmax:
            return torch.argmax(pred)
        return pred


dataset = pd.read_csv('/home/alexandre/dataset/IDMT_guitar_mono_CUT_22050/out.csv', index_col=0)
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

with open("/home/alexandre/logs/classif4july/guitar_Mono/version_1/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)


scaler_mod = torch_clf.ScalerModule()
scaler_mod.mean = scaler.mean
scaler_mod.std = scaler.std
extractor = torch_clf.FeatureExtractor()
cutter = torch_clf.SilenceRemover(0.05, 0.95)
pipe = ClassifierPipeline(clf, extractor, scaler_mod, cutter, with_argmax=False)


example = torch.randn((1, 44100))
p = torch.jit.trace(pipe, example)

torch.jit.save(p, "/home/alexandre/logs/classif4july/guitar_Mono/version_1/pipe_noArgmax.pt")
