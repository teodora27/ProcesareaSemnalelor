import numpy as np
import matplotlib.pyplot as plt

A = 1
f = 5
fs = 100
ts = 1 / fs
t = np.linspace(0, 1, int(fs))

phases = [0, np.pi/4, np.pi/2, np.pi]
signals = [A * np.sin(2 * np.pi * f * t + phi) for phi in phases]

plt.figure(figsize=(10, 6))
for i, signal in enumerate(signals):
    plt.plot(t, signal, label=f'Faza = {phases[i]:.2f} rad')

plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

x = signals[0]
z = np.random.normal(0, 1, len(x))

# normele
norm_x = np.linalg.norm(x)
norm_z = np.linalg.norm(z)

snr_values = [0.1, 1, 10, 100]
noisy_signals = []

# aflam gamma necesare pentru a avea SNR din snr_values
for snr in snr_values:
    gamma = norm_x / (np.sqrt(snr) * norm_z)
    x_noisy = x + gamma * z
    noisy_signals.append(x_noisy)

plt.figure(figsize=(10, 6))
for i, x_noisy in enumerate(noisy_signals):
    plt.plot(t, x_noisy, label=f'SNR = {snr_values[i]}')
plt.title('Semnal sinusoidal cu zgomot Gaussian')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
