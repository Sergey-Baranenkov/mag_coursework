import numpy as np
import torch


def evaluate_multilabel(device, models, matrix_shape, dataloaders, instrument_to_idx):
    true_matrix = np.empty(matrix_shape)
    true_matrix[:] = np.nan
    pred_matrix = np.empty(matrix_shape)
    pred_matrix[:] = np.nan

    for instrument, dataloader in dataloaders.items():
        instr_idx = instrument_to_idx[instrument]

        model = models[instrument]
        model.eval()
        with torch.no_grad():
            for x_data, y_true, indices in dataloader:
                x_data, y_true = x_data.to(device), y_true.to(device)
                pred_logits = model(x_data)
                y_pred = (torch.sigmoid(pred_logits) > 0.5).flatten()
                true_matrix[indices, instr_idx] = y_true.cpu().numpy()
                pred_matrix[indices, instr_idx] = y_pred.cpu().numpy()

    return true_matrix, pred_matrix
