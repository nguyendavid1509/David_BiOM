'''
Deliverable 1: Integrate Audio Input with the Processing Pipeline
- Purpose: move from simply being able to read an audio file on the Pi to establishing the actual audio input interface 
            that the Software-Integration pipeline will use.  
- Pipeline: audio source -> AUDIO_INPUT_MODULE -> audio data -> chunking/processing 

- The exact implementation of the audio input module should reflect what Electronics confirms about the actual audio interface. 

'''

import numpy as np


''' Function: receive_audio
- Purpose: fills the buffer array with incoming audio 
- Inputs: 
    - audio_packet: incoming 960-byte audio packets from the ESP32
    - packet_size: number of bytes in each packet 
    - max_buffer_duration: time of the preprocessing window 
    - audio_buffer: mutable byte buffer sequence to accumulate incoming audio packets   
- Outputs:  
    - full_flag: true when enough audio has been received for processing 
    - audio: Numpy array containing int16 audio samples when the buffer is full, otherwise None
'''
def receive_audio(audio_packet, packet_size, max_buffer_duration, audio_buffer):     
    # packets:
    # packet 0: [A A A A ...]  ← 960 bytes
    # packet 1: [B B B B ...]  ← 960 bytes
    # packet 2: [C C C C ...]  ← 960 bytes

    # audio_buffer:
    # [A A A A ... A][B B B B ... B][C C C C ... C]
    # └──── 960 ─────┘└──── 960 ─────┘└──── 960 ─────┘
    # len(audio_buffer) for a 10ms window (1000 packets) will be 960,000 bytes

    # the incoming audio are byte objects; extend to the current buffer sequence 
    audio_buffer.extend(audio_packet)

    # if the byte_buffer is full, convert the bytes to int16 and reconstruct 
    max_bytes = max_buffer_duration / (10 * (10**(-3))) * packet_size    # number of bytes corresponding to desired buffer window

    # actions to do once the buffer is full
    if len(audio_buffer) >= max_bytes:  
        # converting to 16-bit combines every 2 bytes -> 480,000 int16 samples
        audio = np.frombuffer(audio_buffer[:max_bytes], dtype=np.int16)

        # the audio also needs to be downsampled from 48 kHz to 24 kHz  
        
        # actions to do
        full_flag = True
        audio_buffer.clear()        
    else: 
        audio = None
        full_flag = False

    return full_flag, audio


