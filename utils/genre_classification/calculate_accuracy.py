import pandas as pd
import torch


def calculate_accuracy(y_pred: pd.Series, y_true: pd.Series):
    correct_pred = y_pred == y_true
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc.item()