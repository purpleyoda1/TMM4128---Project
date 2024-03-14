import librosa
import numpy as np 
import random
import pandas as pd 
import joblib
from sklearn.preprocessing import StandardScaler
import warnings

def extract_from_audio(filepath):

    #Load data
    y, sr = librosa.load(filepath)

    #Check and adjust duration
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < 30:
        raise ValueError("Audio file too short, must be 30 seconds")
    
    start = random.randint(0, int(duration) - 30)
    end = start + 30
    y = y[start * sr:end * sr]

    #Extract data
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    harmony = librosa.effects.harmonic(y=y)
    perceptr = librosa.feature.spectral_contrast(y=y, sr=sr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        tempo = librosa.beat.tempo(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    #Fit data to the needed values
    features = {
        'chroma_stft_mean': np.mean(chroma_stft),
        'chroma_stft_var': np.var(chroma_stft),
        'rms_mean': np.mean(rms),
        'rms_var': np.var(rms),
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_var': np.var(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_var': np.var(spectral_bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_var': np.var(rolloff),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
        'zero_crossing_rate_var': np.var(zero_crossing_rate),
        'harmony_mean': np.mean(harmony),
        'harmony_var': np.var(harmony),
        'perceptr_mean': np.mean(perceptr),
        'perceptr_var': np.var(perceptr),
        'tempo': tempo[0],
        'mfcc1_mean': np.mean(mfcc[0]),
        'mfcc1_var': np.var(mfcc[0]),
        'mfcc2_mean': np.mean(mfcc[1]),
        'mfcc2_var': np.var(mfcc[1]),
        'mfcc3_mean': np.mean(mfcc[2]),
        'mfcc3_var': np.var(mfcc[2]),
        'mfcc4_mean': np.mean(mfcc[3]),
        'mfcc4_var': np.var(mfcc[3]),
        'mfcc5_mean': np.mean(mfcc[4]),
        'mfcc5_var': np.var(mfcc[4]),
        'mfcc6_mean': np.mean(mfcc[5]),
        'mfcc6_var': np.var(mfcc[5]),
        'mfcc7_mean': np.mean(mfcc[6]),
        'mfcc7_var': np.var(mfcc[6]),
        'mfcc8_mean': np.mean(mfcc[7]),
        'mfcc8_var': np.var(mfcc[7]),
        'mfcc9_mean': np.mean(mfcc[8]),
        'mfcc9_var': np.var(mfcc[8]),
        'mfcc10_mean': np.mean(mfcc[9]),
        'mfcc10_var': np.var(mfcc[9]),
        'mfcc11_mean': np.mean(mfcc[10]),
        'mfcc11_var': np.var(mfcc[10]),
        'mfcc12_mean': np.mean(mfcc[11]),
        'mfcc12_var': np.var(mfcc[11]),
        'mfcc13_mean': np.mean(mfcc[12]),
        'mfcc13_var': np.var(mfcc[12]),
        'mfcc14_mean': np.mean(mfcc[13]),
        'mfcc14_var': np.var(mfcc[13]),
        'mfcc15_mean': np.mean(mfcc[14]),
        'mfcc15_var': np.var(mfcc[14]),
        'mfcc16_mean': np.mean(mfcc[15]),
        'mfcc16_var': np.var(mfcc[15]),
        'mfcc17_mean': np.mean(mfcc[16]),
        'mfcc17_var': np.var(mfcc[16]),
        'mfcc18_mean': np.mean(mfcc[17]),
        'mfcc18_var': np.var(mfcc[17]),
        'mfcc19_mean': np.mean(mfcc[18]),
        'mfcc19_var': np.var(mfcc[18]),
        'mfcc20_mean': np.mean(mfcc[19]),
        'mfcc20_var': np.var(mfcc[19])
    }

    X_df = pd.DataFrame([features])


    #Scaling drit jeg ikke får til å funke // trengs tydeligvis ikke, la det stå til minne om timene kastet bort
    """ training_data,_,_= data_loader.data_loader('Datasets/Music_genre/features_30_sec.csv')
    scaler = StandardScaler()
    scaler.fit(training_data)
    print(f"Scaler mean: {scaler.mean_}\nScaler scale: {scaler.scale_}")

    scaler = joblib.load(scaler_path)
    X = X.reshape(1, -1)
    X = scaler.transform(X) """

    return X_df