import torch

from .calculate_acurracy import calculate_accuracy


def evaluate(device, model, dataloader, criterion, return_pred: bool):
    loss_sum = 0
    accuracy_sum = 0

    pred_global = []
    true_global = []

    model.eval()
    with torch.no_grad():
        for x_data, y_true in dataloader:
            x_data, y_true = x_data.to(device), y_true.to(device)
            pred_logits = model(x_data)
            batch_loss = criterion(pred_logits, y_true).item()
            loss_sum += batch_loss

            y_pred_softmax = torch.log_softmax(pred_logits, dim=1)
            _, y_pred = torch.max(y_pred_softmax, dim=1)

            accuracy_sum += calculate_accuracy(y_pred, y_true)

            if return_pred:
                pred_global.append(y_pred)
                true_global.append(y_true)

    return loss_sum / len(dataloader), accuracy_sum / len(dataloader), \
        (torch.cat(pred_global), torch.cat(true_global)) if return_pred else (torch.tensor([]).to(device), torch.tensor([]).to(device))
