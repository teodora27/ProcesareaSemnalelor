import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, signal = wavfile.read('vocale.wav') 
N = len(signal)

# daca sunt mai multe canale pastram doar unul
if signal.ndim == 2:
    signal = signal[:, 0]  

group_size = N // 100
step_size = group_size // 2
num_groups = (N - group_size) // step_size + 1

spectrogram = []
for i in range(num_groups):
    start = i * step_size
    end = start + group_size
    segment = signal[start:end]
    fft_vals = np.fft.fft(segment)
    fft_magnitude = np.abs(fft_vals[:group_size // 2]) 
    # folosim prima jumatate, a doua jumatate e simatrica
    # folosim abs pentru a calcula modulul fiecarui nr complex
    spectrogram.append(fft_magnitude)

spectrogram_matrix = np.array(spectrogram).T
spectrogram_db = 20 * np.log10(spectrogram_matrix + 1e-6)  # evită log(0)

plt.figure(figsize=(10, 6))
plt.imshow(spectrogram_db, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label='Amplitudine (dB)')
plt.xlabel('Grupuri de timp')
plt.ylabel('Frecventa')
plt.tight_layout()
plt.show()
