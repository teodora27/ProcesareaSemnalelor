import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

fs = 1000        
duration = 1     
t = np.linspace(0, duration, int(fs * duration))

f_sin = 5         
x_sin = np.sin(2 * np.pi * f_sin * t)
f_saw = 5         
x_saw = sawtooth(2 * np.pi * f_saw * t)

x_sum = x_sin + x_saw

fig, axs = plt.subplots(3, figsize=(10, 6))
axs[0].plot(t, x_sin)
axs[0].set_title("Semnal sinusoidal")
axs[1].plot(t, x_saw)
axs[1].set_title("Semnal sawtooth")
axs[2].plot(t, x_sum)
axs[2].set_title("Suma semnalelor")

for ax in axs:
    ax.grid(True)
    ax.set_xlim([0, duration])

plt.tight_layout()
plt.show()
