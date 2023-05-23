import librosa


def predict_bpm(filename):
    # Загружаем файл
    y, sr = librosa.load(filename)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    return tempo
