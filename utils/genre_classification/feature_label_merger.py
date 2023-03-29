import numpy as np


class FeatureLabelMerger:
    @staticmethod
    def merge(id_to_feature: dict[str, any], id_to_label: dict[str, any]) -> np.ndarray:
        merged = []

        for key in id_to_label.keys():
            if key in id_to_feature:
                merged.append((id_to_feature[key], id_to_label[key]))

        return np.array(merged, dtype=object)
