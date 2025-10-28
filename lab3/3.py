import numpy as np
import matplotlib.pyplot as plt

fs = 1000         
duration = 1       
N = int(fs * duration)
n = np.arange(N)

f1 = 50
f2 = 120
f3 = 300

x = np.sin(2 * np.pi * f1 * n / fs) + \
    0.5 * np.sin(2 * np.pi * f2 * n / fs) + \
    0.3 * np.sin(2 * np.pi * f3 * n / fs)

X = np.zeros(N, dtype=complex)
for k in range(N):
    for m in range(N):
        X[k] += x[m] * np.exp(-2j * np.pi * k * m / N)

freqs = np.arange(N) * fs / N
magnitude = np.abs(X)

plt.figure(figsize=(10, 5))
plt.plot(freqs[:N//2], magnitude[:N//2], color='purple')
plt.xlabel("Frecventa (Hz)")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.tight_layout()
plt.show()
