'''
Deliverable 1: Integrate Audio Input with the Processing Pipeline
- Purpose: move from simply being able to read an audio file on the Pi to establishing the actual audio input interface 
            that the Software-Integration pipeline will use.  
- Pipeline: audio source -> AUDIO_INPUT_MODULE -> audio data -> chunking/processing 

- Should not need to know whether the audio came from a USB device, stored WAV file, or another input source. 
- The exact implementation of the audio input module should reflect what Electronics confirms about the actual audio interface. 

'''

import os 
import librosa 
from pathlib import Path

AUDIO_FORMAT = [".wav", ".mp3", ".flac"]  # Add more formats as needed


''' Function: import_audio
- Purpose: loads a specified audio file or all audio files in the folder and returns the raw audio data (nChannels, nSamples) and sample rate.
- Inputs: path of the folder, optional file name (if not provided, loads all audio files in the folder) 
- Outputs: audio data and sample rate for a single file, or a dictionary of audio data and sample rates for all files in the folder.
'''
def import_audio(folder_path, file_name=None): 

    # Load a specified audio file  
    if file_name is not None: 
        file_path = os.path.join(folder_path, file_name)   # Use the default sample audio file
        audio, sample_rate = librosa.load(file_path, mono=True, sr=24000)

        return audio, sample_rate

    # Load all audio files in the folder 
    else:
        audio_files = {}    # dictionary to hold audio data and sample rates for all files in the folder

        # Loop through all files in the folder (strings) and load audio data for each file
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in AUDIO_FORMAT:   # Path object allows suffix checking 

                # Load the audio file: store the audio data and sample rate in the dictionary 
                file_path = os.path.join(folder_path, file) 
                audio, sample_rate = librosa.load(file_path, mono=True, sr=24000)

                # each entry in the dictionary is a tuple of (audio data, sample rate) where the key is the file name 
                audio_files[file] = (audio, sample_rate)    

        return audio_files


### Diagnostic
cat_audio, cat_sample_rate = import_audio("data", "cat_1.wav")  # Use the default sample audio file
print(f'Cat audio: shape = {cat_audio.shape}, sample rate = {cat_sample_rate}, duration = {cat_audio.shape[1] / cat_sample_rate:.2f} seconds, n_Channels = {cat_audio.shape[0]}')  # Print the audio data and sample rate for the loaded audio files

# audio_files = import_audio("data")  # Load all audio files in the folder
# print(f'Keys in the dictionary: {list(audio_files.keys())}')  # Print the audio data and sample rate for the loaded audio files
# print(f'Audio data and sample rate for cat_1.wav: {audio_files["cat_1.wav"]}')  # Print the audio data and sample rate for the loaded audio files
# print(f'Audio data and sample rate for penguin_1.wav: {audio_files["penguin_1.wav"]}')  # Print the audio data and sample rate for the loaded audio files

