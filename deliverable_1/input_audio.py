import os
import numpy as np
import matplotlib.pyplot as plt
import librosa 
from pathlib import Path
from settings import SAMPLE_FREQUENCY, AUDIO_FORMAT, FOLDER_PATH


''' Function: import audio
Purpose: returns the raw audio data from a user-specified audio file location
Input: user specifies a folder path and/or specific file name to the desired audio file
Output: raw audio data and sample rate 
'''
def import_audio(folder_path, file_name=None): 
    
    # Store the audio files as a dictionary
    audio_files = {}
    
    # Load a specified audio file  
    if file_name is not None: 
        file_path = os.path.join(folder_path, file_name)   # Use the default sample audio file
        audio, sample_rate = librosa.load(file_path, mono=True, sr=SAMPLE_FREQUENCY)

        # Create a new entry in the dictionary to store the (audio data, sample_rate)
        audio_files[file_name] = (audio, sample_rate)


    # Load all audio files in the folder 
    else:
        audio_files = {}    # dictionary to hold audio data and sample rates for all files in the folder

        # Loop through all files in the folder (strings) and load audio data for each file
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in AUDIO_FORMAT:   # Path object allows suffix checking 

                # Load the audio file: store the audio data and sample rate in the dictionary 
                file_path = os.path.join(folder_path, file) 
                audio, sample_rate = librosa.load(file_path, mono=True, sr=SAMPLE_FREQUENCY)

                # Create a new entry in the dictionary for (audio data, sample rate) where the key is the file name 
                audio_files[file] = (audio, sample_rate)    

    return audio_files



''' Function: visualise_audio
Purpose: plotting raw audio data 
'''
def visualise_audio(audio_data): 
    
    files = list(audio_data.keys())

    for file in files: 

        raw_data = audio_data[file][0]
        sample_rate = audio_data[file][1]

        # Create x axis (option for sample or time) 
        sample = np.arange(len(raw_data))
        time = sample / sample_rate

        # Plotting     
        plt.figure()
        plt.ylabel("Amplitude")
        plt.title(f'{file} raw data')
        plt.tight_layout()

        # # Plot in sample domain 
        # plt.xlabel("Sample")
        # plt.plot(sample, raw_data)                

        # Plot in time domain
        plt.xlabel("Time (s)")
        plt.plot(time, raw_data)
        
        plt.show()


# ### Diagnostic
# audio_data = import_audio(FOLDER_PATH)
# visualise_audio(audio_data)
