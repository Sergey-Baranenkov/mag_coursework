from typing import List

import numpy as np
from scipy.stats import skew, kurtosis

from . import AudioPreprocessor


class Extractor:
    def execute(self, audio: AudioPreprocessor, hop_size: int, win_length: int):
        return None


# Сжать 2d фичи в 1 вектор (mean, std, kurtosis, skewness каждой фичи)
def feature_flattener(features: np.ndarray):
    stacked = np.vstack((
        np.mean(features, axis=0),
        np.std(features, axis=0),
        skew(features, axis=0),
        kurtosis(features, axis=0))
    )
    # Если kurtosis или skewness дают nan из-за 0 std - заменяем нулями
    stacked[np.isnan(stacked)] = 0.0

    return stacked.flatten()


# Класс, формирующий фичи аудиозаписи
class FeatureExtractor:
    def __init__(self,
                 audio: AudioPreprocessor,
                 extractors: List[Extractor],
                 window_size_ms: int,
                 hop_size_ms: int,
                 flatten_features: bool,
                 sample_rate: int = 22050,
                 ):
        self.audio = audio

        # Размер окна в кол-ве семплов
        self.window_size = int(window_size_ms * 0.001 * sample_rate)
        # Шаг в кол-ве семплов
        self.hop_size = int(hop_size_ms * 0.001 * sample_rate)

        self.extractors = extractors
        self.flatten_features = flatten_features

    def get_features(self):
        features = []
        for extractor in self.extractors:
            feature = extractor.execute(self.audio, self.hop_size, self.window_size)
            features.append(feature)

        resulting_features = np.hstack(features)
        return resulting_features if self.flatten_features is False else feature_flattener(resulting_features)
