import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

X = misc.face(gray=True)
Y = np.fft.fft2(X)

signal_energy = np.sum(np.abs(Y)**2)

SNR_target_db = 20
SNR_target = 10**(SNR_target_db/10)

# sortam dupa magnitudine
magnitudes = np.abs(Y).flatten()
sorted_indices = np.argsort(magnitudes)[::-1]

signal_power = np.cumsum(magnitudes[sorted_indices]**2)

signal_power = 0.0
cutoff_index = 0
for i, idx in enumerate(sorted_indices):
    signal_power += magnitudes[idx]**2
    noise_energy = signal_energy - signal_power
    if noise_energy == 0:
        cutoff_index = i
        break
    current_SNR = signal_power / noise_energy
    if current_SNR >= SNR_target:
        cutoff_index = i
        break

# masca cu coef pastrati
mask = np.zeros_like(magnitudes, dtype=bool)
mask[sorted_indices[:cutoff_index]] = True
mask = mask.reshape(Y.shape)

Y_compressed = np.where(mask, Y, 0)

X_compressed = np.fft.ifft2(Y_compressed)
X_compressed = np.real(X_compressed)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(X, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(X_compressed, cmap='gray')
plt.title(f'Compresie (SNR {SNR_target_db} dB)')
plt.axis('off')
plt.show()
