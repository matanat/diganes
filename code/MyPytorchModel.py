import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import numpy as np
from sklearn.metrics import f1_score

class PretrainedClassifier(nn.Module):
    def __init__(self, pretrained_model, out_labels, layers_to_freeze=10):
        super().__init__()

        self.feature_extractor = pretrained_model.features

        # Freeze the parameters of the low-level convolutional bottlenecks
        for param in self.feature_extractor[0:layers_to_freeze].parameters():
            param.requires_grad = False

        # Pooling is reliant on the input image size, e.g. for size 64 => (2, 2).
        self.avg_pool = nn.AvgPool2d((7, 7))

        self.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=out_labels, bias=True),
                                        nn.Sigmoid())

        self.init_weights(self.classifier)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.feature_extractor(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x

class MyPytorchModel(pl.LightningModule):

    def __init__(self, hparams, dataset, pretrained_model):
        super().__init__()

        self.hparams = hparams
        self.model = PretrainedClassifier(pretrained_model, out_labels=len(dataset.classes))
        self.dataset = self.init_datasets(dataset)

    def init_datasets(self, dataset, train_per=0.8, val_per=0.1):
        #prepare dataset split
        N = len(dataset)
        train_size = int(N * train_per)
        val_size = int(N * val_per)
        test_size = N - (train_size + val_size)

        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        #init pos_weights of loss function
        pos_count = dataset.label_count()
        pos_weight = (len(dataset) - pos_count) / pos_count

        return {"train": train_set, "val": val_set, "test": test_set}

    def forward(self, x):
        x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):

        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.binary_cross_entropy(out, targets)

        # simplw tresholding at the moment
        preds = (out.data > 0.5).float()

        # macro-f1 instead of acc
        f_score = torch.tensor(f1_score(targets, preds, average='macro', zero_division=0))

        return loss, f_score

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        avg_f1 = torch.stack([x[mode + '_f1_score'] for x in outputs]).mean()

        return avg_loss, avg_f1

    def training_step(self, batch, batch_idx):
        loss, f_score = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_f1_score':f_score, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, f_score = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_f1_score':f_score}

    def test_step(self, batch, batch_idx):
        loss, f_score = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_f1_score':f_score}

    def validation_end(self, outputs):
        avg_loss, f_score = self.general_end(outputs, "val")
        print("Val-F1={:.2f}".format(f_score))
        tensorboard_logs = {'val_loss': avg_loss, 'val_f1': f_score}
        return {'val_loss': avg_loss, 'val_f1': f_score, 'log': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["lr"])
        return optim

    def getDataloaderF1(self, loader = None):
        self.model.eval()

        if not loader: loader = self.val_dataloader()

        scores = []
        labels = []
        for batch in loader:
            X, y = batch

            score = self.forward(X)
            preds = (score.data > 0.5).float()

            scores.append(preds.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        f_score = f1_score(labels, scores, average='macro', zero_division=0)
        return f_score, scores, labels
