import numpy as np
import matplotlib.pyplot as plt

fs = 1000          
duration = 1      
t = np.linspace(0, duration, int(fs * duration))
A = 1             
phi = 0          

# f = fs / 2
f_a = fs / 2
x_a = A * np.sin(2 * np.pi * f_a * t + phi)

#f = fs / 4
f_b = fs / 4
x_b = A * np.sin(2 * np.pi * f_b * t + phi)

# f = 0 Hz
f_c = 0
x_c = A * np.sin(2 * np.pi * f_c * t + phi)

fig, axs = plt.subplots(3, figsize=(10, 6))

axs[0].plot(t, x_a)
axs[0].set_title(f"(a) f = fs/2 = {f_a} Hz")

axs[1].plot(t, x_b)
axs[1].set_title(f"(b) f = fs/4 = {f_b} Hz")

axs[2].plot(t, x_c)
axs[2].set_title(f"(c) f = 0 Hz")

for ax in axs:
    ax.set_xlim([0, 1])  
    ax.grid(True)

plt.tight_layout()
plt.show()
