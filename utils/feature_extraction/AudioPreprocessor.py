# Класс, для загрузки и препроцессинга аудио
import warnings

import librosa
from librosa.util import fix_length

class AudioPreprocessor:
    def __init__(self, path: str, sample_rate = 22050, preserve_sec = 15, shift_sec = 5):
        # Загружаем аудиозапись, указываем схлопывание стерео в моно и down/upsampling до sample_rate
        song, _ = librosa.load(path, mono=True, sr = sample_rate, res_type="kaiser_best", offset = shift_sec, duration = preserve_sec)

        # Если аудио меньше чем shift_sec + preserve_sec по длительности - делаем паддинг и печатаем ворнинг
        desired_sample_num = int(sample_rate * preserve_sec)
        pad_by = desired_sample_num - song.shape[0]

        # Если не хватает больше 10% - выбрасываем ошибку. Нужно отбросить такое аудио
        if pad_by > desired_sample_num / 10 :
            raise Exception(f'Song duration is {song.shape[0] / sample_rate} that is more than 10 times smaller than {preserve_sec}')

        if pad_by != 0:
            warnings.warn(f'Padding audio {path} by {pad_by} samples')
            song = fix_length(song, size=desired_sample_num)

        self.duration = preserve_sec
        self.path = path
        self.sample_rate = sample_rate
        self.song = song