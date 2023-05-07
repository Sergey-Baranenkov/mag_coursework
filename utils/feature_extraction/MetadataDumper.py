# Класс, сохраняющий метаданные
import os
import pickle
from typing import List

from pqdm.processes import pqdm

from . import FeatureExtractor
from . import AudioPreprocessor


def get_id_from_filename(filename: str) -> str:
    return filename.split('.')[0].lstrip('0')


class MetadataDumper:
    def __init__(self,
                 in_dir_path: str,
                 out_path: str,
                 kwargs_for_preprocessor: dict = None,
                 kwargs_for_extractor: dict = None,
                 filter_ids: List[int] = None,
                 ):
        if kwargs_for_extractor is None:
            kwargs_for_extractor = {}
        if kwargs_for_preprocessor is None:
            kwargs_for_preprocessor = {}

        self.kwargs_for_extractor = kwargs_for_extractor
        self.kwargs_for_preprocessor = kwargs_for_preprocessor

        self.in_dir_path = in_dir_path
        self.out_path = out_path
        self.filter_ids = filter_ids

    def func(self, args):
        path, name = args
        full_path = os.path.join(path, name)
        id = get_id_from_filename(name)
        audio = AudioPreprocessor(full_path, **self.kwargs_for_preprocessor)

        extractor = FeatureExtractor(audio, **self.kwargs_for_extractor)
        features = extractor.get_features()

        return id, features

    def execute(self, n_jobs = 4, n_files: int = None):

        paths = [(path, name) for path, subdirs, files in os.walk(self.in_dir_path) for name in files]
        if self.filter_ids is not None:
            paths = list(filter(lambda x: int(get_id_from_filename(x[1])) in self.filter_ids, paths))

        if n_files is not None:
            paths = paths[:n_files]

        rows_of_pairs = pqdm(paths, self.func, n_jobs)
        # Иногда могут возникать ошибки на битых аудио. Отбросим такие аудио
        rows_of_pairs = [val for val in rows_of_pairs if isinstance(val, tuple)]

        print(f'После преобразования {len(paths)} осталось {len(rows_of_pairs)} аудио. Ошибок {len(paths) - len(rows_of_pairs)}')
        pickle.dump(dict(rows_of_pairs), open(self.out_path, 'wb'))