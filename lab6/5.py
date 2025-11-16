import numpy as np
import matplotlib.pyplot as plt

def fereastra_dreptunghiulara(N):
    return np.ones(N)

def fereastra_hanning(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / N))

Nw = 200
f = 100
A = 1
phi = 0

n = np.arange(Nw)
t = n / Nw   

x = A * np.sin(2 * np.pi * f * t + phi)

x_rect = x * fereastra_dreptunghiulara(Nw)
x_hann = x * fereastra_hanning(Nw)

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(n, x, color='green')
plt.title('Sinusoida initiala')
plt.xlabel('n')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(n, x_rect)
plt.title('Semnal prin fereastra dreptunghiulara')
plt.xlabel('n')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(n, x_hann, color='orange')
plt.title('Semnal trecuta prin fereastra Hanning')
plt.xlabel('n')
plt.ylabel('Amplitudine')
plt.grid(True)

plt.tight_layout()
plt.show()
