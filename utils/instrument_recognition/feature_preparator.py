import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from . import InstrumentClassificationDataset

def feature_preparator(batch_size: int, positive_threshold = 0.5, transform=None):
    # Загружаем предвычисленные фичи
    OPENMIC = np.load('data/openmic-2018/openmic-2018.npz', allow_pickle=True)
    X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

    # Загружаем словарь-маппер инструментов
    with open('data/openmic-2018/class-map.json', 'r') as f:
        instrument_to_idx = json.load(f)

    idx_to_instrument = { value:key for key, value in instrument_to_idx.items()}

    # Создатели open-mic-2018 разделили на тренировочную и тестовые выборки по 14915 и 5085 соответственно
    split_train = set(pd.read_csv('data/openmic-2018/partitions/split01_train.csv',
                                  header=None).squeeze())

    split_test = set(pd.read_csv('data/openmic-2018/partitions/split01_test.csv',
                                 header=None).squeeze())

    train_set = set(split_train)
    test_set = set(split_test)

    idx_train, idx_test = [], []

    for idx, n in enumerate(sample_key):
        if n in train_set:
            idx_train.append(idx)
        elif n in test_set:
            idx_test.append(idx)
        else:
            raise RuntimeError('Unknown sample key={}! Abort!'.format(sample_key[n]))

    # Получаем индексы, по которым будем делить данные
    idx_train = np.asarray(idx_train)
    idx_test = np.asarray(idx_test)

    # idx_test составляет примерно 25% от всей выборки. Разделим на 10% для валидации и 15% для теста
    idx_val, idx_test = np.split(idx_test, [int(0.4 * len(idx_test))])

    # Sample[n] : timestamp[10] : features[128]
    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]

    # Sample[n] : Уверенности в наличие инструмента[20]
    Y_true_train = Y_true[idx_train]
    Y_true_val = Y_true[idx_val]
    Y_true_test = Y_true[idx_test]

    # Sample[n] : Известно ли наличие инструмента[20]
    Y_mask_train = Y_mask[idx_train]
    Y_mask_val = Y_mask[idx_val]
    Y_mask_test = Y_mask[idx_test]

    dataloaders = {}

    for instrument in instrument_to_idx:
        # Выбираем id инструмента
        inst_num = instrument_to_idx[instrument]

        # Получаем маску для инструментов
        train_inst = Y_mask_train[:, inst_num]
        val_inst = Y_mask_val[:, inst_num]
        test_inst = Y_mask_test[:, inst_num]

        # Выбираем только те x и y, в которых было измерено наличие или отсутствие инструмента
        X_train_inst = X_train[train_inst]
        X_val_inst = X_val[val_inst]
        X_test_inst = X_test[test_inst]

        # Если confidence в инструменте >0.5, то считаем что он есть в песне
        Y_true_train_inst = Y_true_train[train_inst, inst_num] >= positive_threshold
        Y_true_val_inst = Y_true_val[val_inst, inst_num] >= positive_threshold
        Y_true_test_inst = Y_true_test[test_inst, inst_num] >= positive_threshold

        train_dataset = InstrumentClassificationDataset(X_train_inst, Y_true_train_inst, transform=transform)
        val_dataset = InstrumentClassificationDataset(X_val_inst, Y_true_val_inst, transform=transform)
        test_dataset = InstrumentClassificationDataset(X_test_inst, Y_true_test_inst, transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

        dataloaders[instrument] = [
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ]

    return dataloaders, idx_to_instrument

#%%
