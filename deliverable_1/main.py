from build_packets import simulate_audio_packets
from input_module import receive_audio
from settings import SAMPLE_FREQUENCY, MAX_CHUNK_DURATION

from queue import Queue
import matplotlib.pyplot as plt
import numpy as np

# containers -------------------------------------------------------------------------------------------------------------------- 
audio_buffer = bytearray()  # create a mutable byte buffer to hold the incoming audio packets
preprocessing_queue = Queue(maxsize=10)  # create a queue to hold the audio chunks for processing
# end containers ----------------------------------------------------------------------------------------------------------------


def main(plot_chunks=False): 
    # simulate the incoming audio packets
    packets = simulate_audio_packets(plot=True)     # packets is a list of byte objects, each representing 10ms of audio data

    # loop through the packets and feed them to the receive_audio function
    for packet in packets:
        receive_audio(packet, audio_buffer, preprocessing_queue)      

    audio_chunks = []  # list to hold the audio chunks for visualisation

    while not preprocessing_queue.empty():
        audio_chunks.append(preprocessing_queue.get())
    
    # for visualisation
    if (plot_chunks == True): 

        plt.figure()
        plt.title(f"Audio chunk duration = {MAX_CHUNK_DURATION} seconds")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # colors for plotting

        for i, audio in enumerate(audio_chunks):
            time = np.arange(len(audio)) / SAMPLE_FREQUENCY + (i * MAX_CHUNK_DURATION)  # time axis for each chunk, offset by the chunk duration
            plt.plot(time, audio, color=colours[i], label=f"Chunk {i+1}")
        
        plt.tight_layout()
        plt.legend()
        plt.show()

    print(f"Number of audio chunks: {len(audio_chunks)}")
    print(f"Length of each audio chunk: {len(audio_chunks[0])} samples")
    print(f"Duration of each audio chunk: {len(audio_chunks[0]) / SAMPLE_FREQUENCY} seconds")
    print(f"Actual total duration of audio chunks: {len(audio_chunks) * len(audio_chunks[0]) / SAMPLE_FREQUENCY} seconds")
    print(f"Expected total duration of audio chunks: {len(audio_chunks) * MAX_CHUNK_DURATION} seconds")

    return 0


if __name__ == "__main__":
    main(plot_chunks=False)
