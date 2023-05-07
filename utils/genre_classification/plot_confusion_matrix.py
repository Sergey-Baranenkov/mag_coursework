import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(test_true, test_pred, idx_to_label, transform_idx_to_label):
    confusion_matrix_df = pd.DataFrame(confusion_matrix(test_true.cpu(), test_pred.cpu())) \
        .rename(columns=idx_to_label, index=idx_to_label)

    sns.heatmap(confusion_matrix_df / np.sum(confusion_matrix_df), annot=True, fmt='.1%')

    print(classification_report(
        y_true=list(map(transform_idx_to_label, test_true.cpu().tolist())),
        y_pred=list(map(transform_idx_to_label, test_pred.cpu().tolist())),
        zero_division=0
    ),
    )
