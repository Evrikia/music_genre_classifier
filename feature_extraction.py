import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    tonnetz_mean = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    rms_mean = np.mean(librosa.feature.rms(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return np.hstack([
        mfcc_mean, mfcc_std,
        chroma_mean,
        centroid_mean, bandwidth_mean, rolloff_mean,
        contrast_mean,
        tonnetz_mean,
        zcr_mean,
        rms_mean,
        tempo
    ])
