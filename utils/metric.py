import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def metric(preds,labels):
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return accuracy_score(labels_flat,pred_flat),\
        precision_score(labels_flat,pred_flat,average='macro'), \
        recall_score(labels_flat,pred_flat,average="macro"), \
        f1_score(labels_flat, pred_flat,average="macro")


def metric_cl(preds, labels, sample_weight):
    return accuracy_score(preds, labels, sample_weight=sample_weight),\
        precision_score(preds, labels, sample_weight=sample_weight, average="macro"), \
        recall_score(preds, labels, sample_weight=sample_weight, average="macro"), \
        f1_score(preds, labels, sample_weight=sample_weight, average="macro")
