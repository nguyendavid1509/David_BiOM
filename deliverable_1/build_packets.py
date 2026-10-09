import wave
import numpy as np
import matplotlib.pyplot as plt

from settings import SAMPLE_FREQUENCY, PACKET_SIZE


''' Function: placeholder for the incoming audio packets
Purpose: combines the clarinet.wav and whistle.wav audio files and construct the byte packets 
'''
def simulate_audio_packets(plot=False): 
    # wave module is used as it can parse the header of the WAV file, leaving the raw PCM audio data 
    with wave.open("data/clarinet.wav", "rb") as wav:
        # print("Channels:", wav.getnchannels())
        # print("Sample width:", wav.getsampwidth(), "bytes")
        # print("Sample rate:", wav.getframerate())
        # print("Number of frames:", wav.getnframes())

        audio_1 = wav.readframes(wav.getnframes())

    with wave.open("data/whistle.wav", "rb") as wav: 
        audio_2 = wav.readframes(wav.getnframes())

    # Combine the audio files 
    combined_audio = audio_1 + audio_2 + audio_1 + audio_2 
    print(f'Combined audio length: {len(combined_audio)} ({len(audio_1)} + {len(audio_2)})')

    # plotting functionalities 
    if (plot == True): 
        combined_array = np.frombuffer(combined_audio, dtype=np.int16)
        time = np.arange(len(combined_array)) / SAMPLE_FREQUENCY

        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Combined audio")
        plt.plot(time, combined_array)
        plt.tight_layout()
        plt.show()


    # splitting the combined audio file into packets 
    packets = [combined_audio[i:i + PACKET_SIZE] for i in range(0, len(combined_audio), PACKET_SIZE)]
    return packets


