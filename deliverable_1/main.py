from build_packets import simulate_audio_packets
from input_module import receive_audio
from settings import SAMPLE_FREQUENCY

from queue import Queue
import matplotlib.pyplot as plt
import numpy as np

# containers -------------------------------------------------------------------------------------------------------------------- 
audio_buffer = bytearray()  # create a mutable byte buffer to hold the incoming audio packets
preprocessing_queue = Queue(maxsize=5)  # create a queue to hold the audio chunks for processing
# end containers ----------------------------------------------------------------------------------------------------------------


def main(): 
    # simulate the incoming audio packets
    packets = simulate_audio_packets(plot=True)     # packets is a list of byte objects, each representing 10ms of audio data

    # loop through the packets and feed them to the receive_audio function
    for packet in packets:
        receive_audio(packet, audio_buffer, preprocessing_queue)      
        # audio is a numpy array of int16 samples when the buffer is full


        if not preprocessing_queue.empty():
            audio = preprocessing_queue.get()  # get the audio chunk from the queue

            # visualise the audio chunk
            plt.figure()
            plt.title("Audio chunk")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.plot(np.arange(len(audio)) / SAMPLE_FREQUENCY, audio)
            plt.tight_layout()
            plt.show()

    return 0


if __name__ == "__main__":
    main()
