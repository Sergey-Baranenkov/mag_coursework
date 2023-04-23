import ast
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler, DataLoader

from . import FeatureLabelMerger, GenreClassificationDataset, FeatureNormalizer


def feature_preparator(feature_file: str, batch_size: int, transform=None):
    # Загружаем предвычесленный файл с фичами
    # Объект где ключ - id трека, значение - массив фичей
    id_to_features = pickle.load(open(feature_file, 'rb'))
    id_to_features_old_len = len(id_to_features.values())

    # Нормализуем каждую фичу чтобы они имели 0 матожидание и единичную дисперсию
    # Получаем массив, где индекс соответствует порядку итерации начального объекта
    normalized_features = FeatureNormalizer.normalize(id_to_features.values())
    # Перезаписываем переменную id_to_features соответственно
    id_to_features = {key: normalized_features[idx] for idx, key in enumerate(id_to_features.keys())}
    assert len(id_to_features.values()) == id_to_features_old_len

    # Загружаем метаданные
    metadata = pd.read_csv('./genre_classification_metadata.csv')

    # Создаем словарь маппер id песни - жанр
    id_to_genre = {str(k): v for k, v in metadata.set_index('track_id').to_dict()['title'].items()}

    # Мержим метки и фичи между собой. Примечательно, что сохраняется только пересечение id, которые есть в обоих словарях.
    merged_feature_label = FeatureLabelMerger.merge(id_to_features, id_to_genre)
    # CrossEntropyLoss принимает метки классов от 0 до C-1, напишем прямое и обратное преобразование меток классов
    # Отсортированный массив начальных меток
    merged_feature_label_unique_labels = sorted(np.unique(merged_feature_label[:, 1]))

    # Прямое и обратное преобразование
    label_to_idx = {label: i for i, label in enumerate(merged_feature_label_unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(merged_feature_label_unique_labels)}

    # Создаем кастомный датасет
    dataset = GenreClassificationDataset(merged_feature_label, label_to_idx, transform=transform)

    # Разбиваем выборку на 3 части - тренировочную, валидационную и тестовую
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE

    assert TRAIN_SIZE + VAL_SIZE + TEST_SIZE == 1

    # Делаем это стратифицированно для уверенности, хотя наши классы сбалансированы, это делать необязательно
    train_idx, val_idx = train_test_split(np.arange(len(merged_feature_label)), train_size=TRAIN_SIZE, shuffle=True,
                                          stratify=merged_feature_label[:, 1], random_state=42)

    val_idx, test_idx = train_test_split(val_idx, train_size=VAL_SIZE / (1 - TRAIN_SIZE), shuffle=True,
                                         stratify=merged_feature_label[val_idx][:, 1], random_state=42)

    # Создаем семплеры
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Создаем даталоадеры
    train_data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler)
    val_data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=val_sampler)
    test_data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=test_sampler)

    return train_data_loader, val_data_loader, test_data_loader, idx_to_label

#%%
