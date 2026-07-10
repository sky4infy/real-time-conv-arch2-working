import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

DURATION = 5  # seconds
SAMPLE_RATE = 16000

print("Recording in 3... 2... 1...")
import time; time.sleep(1)
print("SPEAK MARATHI NOW for 5 seconds...")

audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
write("marathi_test.wav", SAMPLE_RATE, audio)
print("Saved to marathi_test.wav")