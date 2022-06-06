import torch
from torch import nn
from source.classifiers.classifier_pytorch import MLPClassifier

CHECKPOINT = "/home/alexandre/classif_logs/torch/version_4/checkpoints/epoch=499-step=38000.ckpt"

clf = MLPClassifier.load_from_checkpoint(CHECKPOINT, input_size=143,
                                         output_size=11, hidden_size=100,
                                         activation='sigmoid', solver='adam',
                                         max_iter=500)


class ClassifierPipeline(nn.Module):
    def __init__(self, classifier, feat_extractor, scaler):
        super(ClassifierPipeline, self).__init__()
        self.clf = classifier
        self.feat_extractor = feat_extractor
        self.scaler = scaler
        self.pipeline = nn.Sequential(
            feat_extractor,
            scaler,
            clf
        )

    def forward(self, audio, rate):
        out = self.pipeline(audio, rate)
        return torch.argmax(out)

