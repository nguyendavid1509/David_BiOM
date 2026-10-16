'''
Deliverable 1: Integrate Audio Input with the Processing Pipeline
- Purpose: move from simply being able to read an audio file on the Pi to establishing the actual audio input interface 
            that the Software-Integration pipeline will use.  
- Pipeline: audio source -> AUDIO_INPUT_MODULE -> audio data -> chunking/processing 

- The exact implementation of the audio input module should reflect what Electronics confirms about the actual audio interface. 

'''

import numpy as np
from settings import MAX_CHUNK_DURATION, PACKET_SIZE, PACKET_DURATION


''' Class: AudioInput 
- Purpose: fills the buffer array with incoming audio 
- Methods: 
    - init: this class owns the audio buffer to turn packets into windows
    - receive_audio: pushes an audio chunk onto the queue for preprocessing if it meets the length requirements 
    - flush: returns the remaining, incomplete audio chunk 
- Inputs: 
    - audio_packet: incoming 960-byte audio packets from the ESP32
    - packet_size: number of bytes in each packet 
    - max_buffer_duration: time of the preprocessing window 
    - audio_buffer: mutable byte buffer sequence to accumulate incoming audio packets   
- Outputs:  
    - full_flag: true when enough audio has been received for processing 
    - audio: Numpy array containing int16 audio samples when the buffer is full, otherwise None
'''

class AudioInput: 

    def __init__(self, preprocessing_queue): 
        self.audio_buffer = bytearray()
        self.preprocessing_queue = preprocessing_queue
        self.max_bytes = int(MAX_CHUNK_DURATION / PACKET_DURATION) * PACKET_SIZE    # number of bytes corresponding to desired buffer window

    def receive_audio(self, audio_packet):     
        # packets:
        # packet 0: [A A A A ...]  ← 960 bytes
        # packet 1: [B B B B ...]  ← 960 bytes
        # packet 2: [C C C C ...]  ← 960 bytes

        # audio_buffer:
        # [A A A A ... A][B B B B ... B][C C C C ... C]
        # └──── 960 ─────┘└──── 960 ─────┘└──── 960 ─────┘
        # len(audio_buffer) for a 10ms window (1000 packets) will be 960,000 bytes

        # the incoming audio are byte objects; extend to the current buffer sequence 
        self.audio_buffer.extend(audio_packet)

        # if the byte_buffer is full, convert the bytes to int16 and reconstruct 

        # actions to do once the buffer is full -> while loop used as the buffer may contain more than one complete window 
        while len(self.audio_buffer) >= self.max_bytes:  
            # converting to 16-bit combines every 2 bytes -> 480,000 int16 samples
            audio = np.frombuffer(
                self.audio_buffer[:self.max_bytes], 
                dtype=np.int16
                ).copy()  # copy to avoid referencing the original buffer
            
            # we have a full packet -> send to preprecessing 
            del self.audio_buffer[:self.max_bytes]    # clears the processed portion of the buffer FUTURE: overlapping portions only need to chop off the front  
            
            # queue the audio chunk for processing      -> NEED A POLICY: what happens if the bounded queue is full?  
            self.preprocessing_queue.put(audio)

    def flush(self): 
        # Discard any incomplete audio from the previous session 
        remaining_audio = bytes(self.audio_buffer)

        # Clear the working buffer 
        self.audio_buffer.clear()
        return remaining_audio
