import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from sklearn.metrics import f1_score

class PretrainedClassifier(nn.Module):
    def __init__(self, pretrained_model, hparams, nof_classes):
        super().__init__()

        self.feature_extractor = pretrained_model.features

        # Freeze the parameters of the low-level convolutional bottlenecks
        for param in self.feature_extractor[0:hparams['layers_to_freeze']].parameters():
            param.requires_grad = False

        # Pooling is reliant on the input image size, e.g. for size 64 => (2, 2).
        self.avg_pool = nn.AvgPool2d((7, 7))

        self.classifier = nn.Sequential(nn.Dropout(hparams['dropout_p']),
                                        nn.Linear(in_features=1280, out_features=nof_classes, bias=True))

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
        self.model = PretrainedClassifier(pretrained_model, hparams,
                                          nof_classes=dataset["train"].nof_classes)

        self.dataset = dataset
        self.criterion = hparams["loss"]

    def forward(self, x):
        x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):

        images, targets = batch

        # forward pass
        logits = self.forward(images)

        # loss
        loss = self.criterion(logits, targets)

        """
        # simple tresholding during training
        preds = (torch.sigmoid(logits).data > 0.5).float()

        # macro-f1 instead of acc
        f_score = torch.tensor(f1_score(targets.detach().cpu().numpy(),
                                        preds.detach().cpu().numpy(),
                                        average='macro', zero_division=0))
        """
        preds = logits.argmax(axis=1)
        n_correct = (targets == preds).sum()

        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        #avg_f1 = torch.stack([x[mode + '_f1_score'] for x in outputs]).mean()

        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])

        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct':n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct':n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'val_n_correct':n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Acc={:.2f}, Val-Loss:{:.2f}".format(acc, avg_loss))
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    #@pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.hparams["batch_size"])

    #@pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"])

    #@pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["lr"],
                                weight_decay=self.hparams["weight_decay"])
        return optim

    def getDataloaderF1(self, loader = None):
        self.model.eval()

        if not loader: loader = self.val_dataloader()

        scores = []
        labels = []
        outputs = []
        for batch in loader:
            X, y = batch

            out = torch.sigmoid(self.forward(X)).data
            preds = (out > 0.5).float()

            scores.append(preds.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            outputs.append(out.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)
        outputs = np.concatenate(outputs, axis=0)

        f_score = f1_score(labels, scores, average='macro', zero_division=0)
        return f_score, scores, labels, outputs
