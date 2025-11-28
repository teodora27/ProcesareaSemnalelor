import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

X = misc.face(gray=True)

pixel_noise = 200
noise = np.random.randint(-pixel_noise, pixel_noise+1, size=X.shape)
X_noisy = X + noise

def compute_snr(original, test):
    signal_energy = np.sum(original**2)
    noise_energy = np.sum((test - original)**2)
    return 10 * np.log10(signal_energy / noise_energy)

snr_before = compute_snr(X, X_noisy)

Y_noisy = np.fft.fft2(X_noisy)
freq_db = 20*np.log10(np.abs(Y_noisy) + 1e-8)

freq_cutoff = 120
Y_filtered = Y_noisy.copy()
Y_filtered[freq_db > freq_cutoff] = 0

X_filtered = np.real(np.fft.ifft2(Y_filtered))
snr_after = compute_snr(X, X_filtered)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(X, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(X_noisy, cmap='gray')
plt.title(f'Noisy (SNR={snr_before:.2f} dB)')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(X_filtered, cmap='gray')
plt.title(f'Filtered (SNR={snr_after:.2f} dB)')
plt.axis('off')

plt.show()
