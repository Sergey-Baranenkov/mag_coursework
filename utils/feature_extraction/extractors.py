from librosa import cqt
from librosa.feature import mfcc, spectral_centroid, spectral_rolloff, spectral_flatness, zero_crossing_rate, \
    chroma_stft, delta, melspectrogram

from utils.feature_extraction import AudioPreprocessor
from utils.feature_extraction.FeatureExtractor import Extractor
import tempfile
import soundfile as sf


class VGGishExtractor(Extractor):
    def __init__(self, model):
        self.model = model

    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        ext = f".{audio.path.split('.')[-1]}"
        # Создаем временный файл для сохранения аудио т.к моделька читает файлы
        with tempfile.NamedTemporaryFile(suffix=ext) as file:
            sf.write(file.name, audio.song, samplerate=audio.sample_rate)
            features = self.model.forward(file.name).cpu().detach().numpy()
            return features


class MFCCExtractor(Extractor):
    def __init__(self, n_mfcc):
        self.n_mfcc = n_mfcc

    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        mfcc_features = mfcc(
            y=audio.song,
            sr=audio.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=hop_size,
            win_length=window_size
        )

        return mfcc_features.T


class MFCCDeltaExtractor(Extractor):
    def __init__(self, n_mfcc, order):
        self.n_mfcc = n_mfcc
        self.order = order

    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        mfcc_features = mfcc(
            y=audio.song,
            sr=audio.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=hop_size,
            win_length=window_size
        )

        return delta(mfcc_features, order=self.order).T


class SpectralCentroidExtractor(Extractor):
    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        spectral_centroid_features = spectral_centroid(
            y=audio.song,
            sr=audio.sample_rate,
            hop_length=hop_size,
            win_length=window_size
        )

        return spectral_centroid_features.T


class SpectralRoloffExtractor(Extractor):
    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        spectral_rolloff_features = spectral_rolloff(
            y=audio.song,
            sr=audio.sample_rate,
            hop_length=hop_size,
            win_length=window_size
        )

        return spectral_rolloff_features.T


class ZCRExtractor(Extractor):
    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        zcr_features = zero_crossing_rate(
            y=audio.song,
            hop_length=hop_size,
        )

        return zcr_features.T


class ChromaExtractor(Extractor):
    def __init__(self, n_chroma: int):
        self.n_chroma = n_chroma

    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        chroma_features = chroma_stft(
            n_chroma=self.n_chroma,
            y=audio.song,
            sr=audio.sample_rate,
            hop_length=hop_size,
            win_length=window_size
        )

        return chroma_features.T


class SpectralFlatnessExtractor(Extractor):
    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        spectral_flatness_features = spectral_flatness(
            y=audio.song,
            hop_length=hop_size,
            win_length=window_size
        )

        return spectral_flatness_features.T


class MelgramExtractor(Extractor):
    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        mel_spec = melspectrogram(
            y=audio.song,
            sr=audio.sample_rate,
            hop_length=hop_size,
            win_length=window_size
        )

        return mel_spec.T


class QTransformationExtractor(Extractor):
    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        cqt_spec = cqt(
            y=audio.song,
            sr=audio.sample_rate,
            hop_length=hop_size
        )

        return cqt_spec.T


class ExternalNNExtractor(Extractor):
    def execute(self, audio: AudioPreprocessor, hop_size: int, window_size: int):
        return audio.path
