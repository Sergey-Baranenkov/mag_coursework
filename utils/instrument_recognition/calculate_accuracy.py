import pandas as pd
import torch


def calculate_accuracy(y_pred: torch.tensor, y_true: torch.tensor):
    y_pred = y_pred.flatten()
    correct_pred = y_pred == y_true
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc.item()