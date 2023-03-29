import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .calculate_acurracy import calculate_accuracy
from .evaluate import evaluate


def executor(device, model, train_dataloader, val_dataloader, epochs, learning_rate=1e-3, evaluate_per_iteration=300,
             weight_decay=0.05, early_stop_after=None) -> (np.array, np.array):
    # Определяем лосс функцию - кроссэнтропия
    criterion = nn.CrossEntropyLoss()  # Можно забацать веса

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # Попробовать weight decay

    train_progress = []  # [(epoch_loss, epoch_accuracy)]
    val_progress = []  # [(epoch_loss, epoch_accuracy)]

    epoch_without_progress = 0
    val_last_epoch_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        train_epoch_loss = 0
        val_epoch_loss = 0

        train_iter_num = 0
        val_iter_num = 0

        train_epoch_accuracy = 0
        val_epoch_accuracy = 0

        for x_data, y_true in train_dataloader:
            train_iter_num += 1

            x_data, y_true = x_data.to(device), y_true.to(device)
            model.train()
            pred_logits = model(x_data)
            optimizer.zero_grad()
            loss = criterion(pred_logits, y_true)
            loss.backward()
            optimizer.step()

            y_pred_softmax = torch.log_softmax(pred_logits, dim=1)
            _, y_pred = torch.max(y_pred_softmax, dim=1)

            train_epoch_loss += loss.item()
            train_epoch_accuracy += calculate_accuracy(y_pred, y_true)

            if train_iter_num % evaluate_per_iteration == evaluate_per_iteration - 1:
                val_iter_num += 1
                val_loss, val_acc, _ = evaluate(device, model, val_dataloader, criterion, return_pred=False)
                val_epoch_loss += val_loss
                val_epoch_accuracy += val_acc

        train_progress.append(
            (train_epoch_loss / max(train_iter_num, 1), train_epoch_accuracy / max(train_iter_num, 1)))
        val_progress.append((val_epoch_loss / max(val_iter_num, 1), val_epoch_accuracy / max(val_iter_num, 1)))

        if early_stop_after is not None:
            max_epoch_without_progress, min_epoch_progress = early_stop_after

            if val_last_epoch_loss - val_progress[epoch][0] < min_epoch_progress:
                epoch_without_progress += 1
            else:
                val_last_epoch_loss = val_progress[epoch][0]
                epoch_without_progress = 0

            if epoch_without_progress >= max_epoch_without_progress:
                print('Early stop!')
                break

    return np.array(train_progress), np.array(val_progress)
