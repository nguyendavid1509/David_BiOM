''' settings.py
Purpose: contains the configuration parameters for the audio input module and the processing pipeline
        current values are based on the AudioMoth specifications and the model input requirements.
'''

# audio configuration parameters
FOLDER_PATH = "data" 
AUDIO_FORMAT = [".wav", ".mp3", ".flac"]  # Add more formats as needed

SAMPLE_FREQUENCY = 48000                    # AudioMoth incoming data 
PACKET_DURATION = 10 * (10**(-3))           # 10 ms of audio data per packet 
PACKET_SIZE = int(SAMPLE_FREQUENCY * PACKET_DURATION * 2)  # in bytes: 10ms of 16-bit audio sampled at 48 kHz = 960 bytes per data transfer 

# chunking parameters 
MAX_CHUNK_DURATION = 5                    # 5 seconds of audio data for the preprocessing window

# model parameters 
MODEL_FREQUENCY = 24000                    # 24 kHz for the model input
