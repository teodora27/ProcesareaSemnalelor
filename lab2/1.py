import numpy as np
import matplotlib.pyplot as plt

A = 1
f = 5
phi = 0
fs = 100  
ts = 1 / fs  
t = np.linspace(0, 1, int(fs)) 

x_sin = A * np.sin(2 * np.pi * f * t + phi)
x_cos = A * np.cos(2 * np.pi * f * t + phi-np.pi/2)

fig, axs = plt.subplots(2)

axs[0].plot(t, x_sin)
axs[0].set_title('Semnal sinusoidal')
axs[0].set_xlim([0, 1])

axs[1].plot(t, x_cos)
axs[1].set_title('Semnal cosinusoidal echivalent')
axs[1].set_xlim([0, 1])

plt.tight_layout()
plt.show()
