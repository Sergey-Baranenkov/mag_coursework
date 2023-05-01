import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(test_true, test_pred):
    confusion_matrix_df = pd.DataFrame(confusion_matrix(test_true.cpu(), test_pred.cpu()))

    sns.heatmap(confusion_matrix_df / np.sum(confusion_matrix_df), annot=True, fmt='.1%')

    print(classification_report(
        y_true=test_true.cpu().tolist(),
        y_pred=test_pred.cpu().tolist(),
    ))
