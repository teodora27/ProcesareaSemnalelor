import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

fs = 16000       
duration = 1    

f1 = 440          
t1 = np.linspace(0, duration, int(fs * duration))
x1 = np.sin(2 * np.pi * f1 * t1)

f2 = 880         
t2 = np.linspace(0, duration, int(fs * duration))
x2 = np.sin(2 * np.pi * f2 * t2)

x_concat = np.concatenate((x1, x2))

sd.play(x_concat, fs)
sd.wait()

plt.figure(figsize=(10, 4))
plt.plot(np.concatenate((t1, t2 + duration)), x_concat)
plt.title("Concatenarea a doua semnale sinusoidale cu frecvente diferite")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")
plt.xlim(0.95, 1.05)  # zoom pe zona de tranzitie
plt.grid(True)
plt.tight_layout()
plt.show()

# Se observa cum sunetul devine mai inalt daca frecventa creste
