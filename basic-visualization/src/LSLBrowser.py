from pylsl import StreamInlet, resolve_streams
import numpy as np
import time

# 1. Find all EEG streams
print("Looking for SynAmpsRT EEG stream...")
streams = resolve_streams('type', 'EEG')  # current function name
inlet = StreamInlet(streams[0])          # connect to first found stream

print("Connected to stream:", streams[0].name())

# 2. Live read loop
while True:
    # Pull a chunk of up to 32 samples
    chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=32)
    if timestamps:
        # chunk is [samples x channels]
        data = np.array(chunk).T  # shape: channels x samples
        print("Chunk shape:", data.shape)
    time.sleep(0.01)
