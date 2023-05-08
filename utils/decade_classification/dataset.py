from typing import List

import numpy as np
from torch.utils.data import Dataset


class DecadeClassificationDataset(Dataset):
    def __init__(self, features_with_labels: List[tuple], label_to_idx: dict, transform=None):
        self.features_with_labels = features_with_labels
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.features_with_labels)

    def __getitem__(self, idx):
        features, label = self.features_with_labels[idx]
        features = features.astype('float32')

        if self.transform:
            features = self.transform(features)

        return features, self.label_to_idx[label]


class DecadeClassificationDatasetForExternal(Dataset):
    def __init__(self, features_with_labels: List[tuple], label_to_idx: dict, transform=None):
        self.features_with_labels = features_with_labels
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.features_with_labels)

    def __getitem__(self, idx):
        features, label = self.features_with_labels[idx]

        if self.transform:
            features = self.transform(features)

        return np.array(features), self.label_to_idx[label]
