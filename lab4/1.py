import numpy as np
import matplotlib.pyplot as plt
import time

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return np.array([even[k] + T[k] for k in range(N // 2)] +
                    [even[k] - T[k] for k in range(N // 2)])

sizes = [128, 256, 512, 1024,2048,4096,8192]

times_dft = []
times_fft = []
times_numpy = []

for N in sizes:
    x = np.random.random(N)

    start = time.time()
    dft(x)
    times_dft.append(time.time() - start)

    start = time.time()
    fft(x)
    times_fft.append(time.time() - start)

    start = time.perf_counter()
    np.fft.fft(x)
    times_numpy.append(time.perf_counter() - start)

print(times_numpy)
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_dft, label='DFT (manual)', marker='o')
plt.plot(sizes, times_fft, label='FFT (manual)', marker='s')
plt.plot(sizes, times_numpy, label='numpy.fft', marker='^')
plt.yscale('log')
plt.xlabel('dimensiune vector')
plt.ylabel('timp')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
