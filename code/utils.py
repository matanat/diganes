import torch
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from .MyPytorchModel import MyPytorchModel

def load_model(model_path, dataset, pretrained_model, loss):
    model_dict = pickle.load(open(model_path, 'rb'))["diganes"]
    model_dict["hparams"]["loss"] = loss
    model = MyPytorchModel(model_dict["hparams"], dataset, pretrained_model)
    model.load_state_dict(model_dict["state_dict"])
    return model

def save_model(model, file_name, directory = "models"):
    model = model.cpu()
    model_dict = {"diganes":{"state_dict":model.state_dict(), "hparams": model.hparams}}
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model_dict, open(os.path.join(directory, file_name), 'wb', 4))

def optimal_tresholds(grid):
    """Comptues the optimal threshold for binary classifcation for each label based on
    F-1 score.
    Recived performance grid as DataFrame.
    """
    max_perf = grid.groupby(['id', 'label', 'freq'])[['f1']].max().sort_values('f1', ascending=False).reset_index()

    nof_classes = len(max_perf)
    thresholds = np.zeros((nof_classes,))
    for i in range(nof_classes):
        max_f1 = max_perf[max_perf.id.eq(i)].f1.values[0]
        thresholds[i] = grid[grid.id.eq(i) & grid.f1.eq(max_f1)].sort_values('threshold', ascending=False).threshold.values[0]

    return thresholds

def perf_grid(outputs, target, label_names, label_freq, n_thresh=100):
    #From https://github.com/ashrefm/multi-label-soft-f1/blob/master/utils.py
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.

    Args:
        X: Images
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        label_freq (list of floats): freq of each label in training set
        n_thresh (int) : number of thresholds to try

    Returns:
        grid (Pandas dataframe): performance table
    """

    # Get predictions
    y_hat_val = outputs

    # Define target matrix
    y_val = target

    # Get label indexes
    label_index = [i for i in range(len(label_names))]
    # Define thresholds
    thresholds = np.linspace(0,1,n_thresh+1).astype(np.float32)

    # Compute all metrics for all labels
    ids, labels, freqs, tps, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], []
    for l in label_index:
        for thresh in thresholds:
            ids.append(l)
            labels.append(label_names[l])
            freqs.append(label_freq[l])
            y_hat = y_hat_val[:,l]
            y = y_val[:,l].astype(int)
            y_pred = (y_hat > thresh).astype(int)
            tp = np.count_nonzero(y_pred  * y)
            fp = np.count_nonzero(y_pred * (1-y))
            fn = np.count_nonzero((1-y_pred) * y)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2*tp / (2*tp + fn + fp + 1e-16)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Create the performance dataframe
    grid = pd.DataFrame({
        'id':ids,
        'label':labels,
        'freq':freqs,
        'threshold':list(thresholds)*len(label_index),
        'tp':tps,
        'fp':fps,
        'fn':fns,
        'precision':precisions,
        'recall':recalls,
        'f1':f1s})

    grid = grid[['id', 'label', 'freq', 'threshold',
                 'tp', 'fn', 'fp', 'precision', 'recall', 'f1']]

    return grid


#taken from https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, y_true,):
        assert logits.ndim == 2
        assert y_true.ndim == 2

        y_true = y_true.to(torch.float32)
        y_pred = torch.sigmoid(logits).to(torch.float32)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        return 1 - f1.mean()
