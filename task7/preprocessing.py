import librosa
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from input import receive_input


# def preprocess_input(data): 
#     # processed_data = [x * 2 for x in data]  # Example preprocessing: doubling each element

#     return processed_data

# '''
# Proof that we have a separate preprocessing stage. 

# Later, it could be something like audio --> buffering --> mel-spectrogram --> model-ready data. 
# '''

def mel_spectrogram(audio, sample_rate, n_mels=128): 
    # Placeholder for mel-spectrogram conversion
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibel (log) scale

    return mel_spec_db  # Return the mel-spectrogram in decibel scale

audio, sample_rate = receive_input("data", "cat_1.wav")
mel_spec_db_cat = mel_spectrogram(audio, sample_rate)  # Use the default sample audio file

# Plot the Mel spectrogram
fig, ax = plt.subplots(figsize=(10, 4))
img = librosa.display.specshow(
    mel_spec_db_cat, x_axis="time", y_axis="mel", sr=sample_rate, ax=ax
)

fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set(title="Mel-frequency spectrogram of cat_1.wav")
plt.show()
