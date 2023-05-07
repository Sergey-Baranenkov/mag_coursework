import numpy as np


def squeeze_answer_matrix(matrix: np.array):
    return list(
        map(
            lambda x: np.array(list(filter(lambda y: not np.isnan(y), x))),
            matrix.tolist()
        )
    )


def get_exact_match_ratio(m_true, m_pred):
    correct = 0
    total = len(m_pred)
    for _pred, _true in zip(m_pred, m_true):
        if np.all(_pred == _true):
            correct += 1

    return correct / total


def get_accuracy(m_true, m_pred):
    acc = 0
    total = len(m_pred)
    for _pred, _true in zip(m_pred, m_true):
        numerator = sum(np.logical_and(_true, _pred))
        denominator = sum(np.logical_or(_true, _pred))
        acc += numerator / denominator if denominator != 0 else 1

    return acc / total
