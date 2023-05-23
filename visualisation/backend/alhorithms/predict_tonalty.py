import librosa
import numpy as np


def krumhansl_schmuckler(filename):
    # Загружаем файл
    y, sr = librosa.load(filename)

    # Удаляем перкуссионные звуки
    y, y_percussive = librosa.effects.hpss(y)

    # Извлекаем хроматические признаки
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=24)
    chroma_mean = np.mean(chroma, axis=1)

    # Определяем полутона
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Объект полутон - среднее
    keyfreqs = {pitches[i]: chroma_mean[i] for i in range(12)}

    # Константы из статьи
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    # Ищем корреляции между минорным и мажорным профилем с каждым циклическим сдвигом полутонов
    maj_corrs = []
    min_corrs = []
    for i in range(12):
        chroma_for_key = [keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
        maj_corrs.append(np.corrcoef(maj_profile, chroma_for_key)[0, 1])
        min_corrs.append(np.corrcoef(min_profile, chroma_for_key)[0, 1])

    # Находим наиболее вероятную мажорную и минорную частоту
    maj_key = np.argmax(maj_corrs)
    min_key = np.argmax(min_corrs)

    # Сравниваем вероятности мажорной и минорной частоты и определяем тональность
    if maj_corrs[maj_key] > min_corrs[min_key]:
        return pitches[maj_key], 'major', maj_corrs[maj_key]
    else:
        return pitches[min_key], 'minor', min_corrs[min_key]
