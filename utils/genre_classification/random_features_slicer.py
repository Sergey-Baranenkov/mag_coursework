import numpy as np


def random_features_slicer(features):
    time_len = features.shape[0]
    slice_len = 256  # чуть больше 5 минут

    min_idx = np.random.randint(time_len - slice_len)

    return features[min_idx:min_idx + slice_len]
