import numpy as np
import matplotlib.pyplot as plt

B = 1
t = np.linspace(-3, 3, 1000)
x = np.sinc(B * t)**2   

Fs_values = [1.0, 1.5, 2.0, 4.0]

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()

for i, Fs in enumerate(Fs_values):
    Ts = 1 / Fs

    n = np.arange(-3/Ts, 3/Ts)
    t_samples = n * Ts
    x_samples = np.sinc(B * t_samples)**2

    t_recon = np.linspace(-3, 3, 1000)
    x_recon = np.zeros_like(t_recon)
    for k in range(len(n)):
        x_recon += x_samples[k] * np.sinc((t_recon - t_samples[k]) / Ts)

    axes[i].plot(t, x, label='Original')
    axes[i].stem(t_samples, x_samples, linefmt='r-')
    axes[i].plot(t_recon, x_recon, 'g--', label='Reconstruit')
    axes[i].set_title(f'Fs = {Fs} Hz')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()
