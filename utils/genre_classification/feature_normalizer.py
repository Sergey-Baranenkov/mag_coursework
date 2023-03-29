from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    @staticmethod
    def normalize(features: List[np.array]) -> List[np.array]:
        initial_len = len(features)
        scaler = StandardScaler()
        features = np.vstack(tuple(features))
        features = scaler.fit_transform(features)
        return np.vsplit(features, initial_len)
