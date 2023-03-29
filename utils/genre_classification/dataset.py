from typing import List

from torch.utils.data import Dataset


class GenreClassificationDataset(Dataset):
    def __init__(self, features_with_labels: List[tuple], label_to_idx: dict):
        self.features_with_labels = features_with_labels
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.features_with_labels)

    def __getitem__(self, idx):
        features, label = self.features_with_labels[idx]

        return features.astype('float32'), self.label_to_idx[label]
