import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# a)
data = pd.read_csv("Train.csv")
counts = data["Count"].values
x = counts[:72]

# b)
windows = [5, 9, 13, 17]

plt.figure(figsize=(12,6))
plt.plot(x, label="Semnal original", color="black")

for w in windows:
    y = np.convolve(x, np.ones(w), 'valid') / w
    plt.plot(range(w-1, w-1+len(y)), y, label=f"Medie alunecatoare w={w}")

plt.title("Filtrare cu medie alunecatoare")
plt.xlabel("Ore")
plt.ylabel("Numar vehicule")
plt.legend()
plt.grid(True)
plt.show()

# c)
# fs = 1/3600 Hz
# fNyquist = 1/(2*fs)=1/7200 Hz
# Alegem perioada de 6 ore cu f = 1/(6*3600) Hz
# frecventa normalizata Wn=f/fNyquist = 0.333

# d)
from scipy.signal import butter, cheby1, filtfilt, freqz

N = 5        # ordinul 
Wn = 0.333   
rp = 5

# Butterworth
b_butter, a_butter = butter(N, Wn, btype='low')
# Chebyshev
b_cheby, a_cheby = cheby1(N, rp, Wn, btype='low')

# e) 
x_butter = filtfilt(b_butter, a_butter, x)
x_cheby = filtfilt(b_cheby, a_cheby, x)

plt.figure(figsize=(12,6))
plt.plot(x, label="Semnal original", color="black")
plt.plot(x_butter, label="Filtru Butterworth", color="blue")
plt.plot(x_cheby, label="Filtru Chebyshev", color="red")
plt.title("Semnal filtrat cu Butterworth si Chebyshev")
plt.xlabel("Ore")
plt.ylabel("Numar vehicule")
plt.legend()
plt.grid(True)
plt.show()

# f) 
orders = [3, 9]        
rp_values = [1, 10]    

plt.figure(figsize=(12,6))
plt.plot(x, label="Semnal original", color="black")

for N in orders:
    b_butter, a_butter = butter(N, Wn, btype='low')
    x_butter = filtfilt(b_butter, a_butter, x)
    plt.plot(x_butter, label=f"Butterworth N={N}")

for rp in rp_values:
    b_cheby, a_cheby = cheby1(5, rp, Wn, btype='low')  # ordin fix 5
    x_cheby = filtfilt(b_cheby, a_cheby, x)
    plt.plot(x_cheby, label=f"Chebyshev rp={rp} dB")

plt.title("Comparatie filtre Butterworth si Chebyshev cu parametri variati")
plt.xlabel("Ore")
plt.ylabel("Numar vehicule")
plt.legend()
plt.grid(True)
plt.show()
