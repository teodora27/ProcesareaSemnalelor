import numpy as np
import matplotlib.pyplot as plt

f0 = 8  #fr semnal
fs = 20  # fr esantion
T = 1  
t_cont = np.linspace(0, T, 1000)  # timp continuu
t_disc = np.arange(0, T, 1/fs)  # timp discret

x0 = np.sin(2 * np.pi * f0 * t_cont)
x1 = np.sin(2 * np.pi * (-2*fs + f0) * t_cont) 
x2 = np.sin(2 * np.pi * (-fs +  f0) * t_cont) 

# esantioane
x0_disc = np.sin(2 * np.pi * f0 * t_disc)
x1_disc = np.sin(2 * np.pi * (-2*fs + f0) * t_disc)
x2_disc = np.sin(2 * np.pi * (-fs + f0) * t_disc)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t_cont, x0, color='blue')
plt.stem(t_disc, x0_disc, linefmt='y-', basefmt='k-')
plt.axhline(0, color='black', linewidth=1.5) 

plt.subplot(3, 1, 2)
plt.plot(t_cont, x1, color='red')
plt.stem(t_disc, x1_disc, linefmt='y-', basefmt='k-')
plt.axhline(0, color='black', linewidth=1.5) 

plt.subplot(3, 1, 3)
plt.plot(t_cont, x2, color='green')
plt.stem(t_disc, x2_disc, linefmt='y-', basefmt='k-')
plt.axhline(0, color='black', linewidth=1.5) 

plt.tight_layout()
plt.show()
