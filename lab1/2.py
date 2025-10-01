import numpy as np
import matplotlib.pyplot as plt

# a
t = np.linspace(0, 1, 1600)  
zoom=100
t = np.linspace(0, 1/zoom, int(1600/zoom))
x = np.sin(400*2*np.pi*t)
fig, axs = plt.subplots(2)

axs[0].plot(t, x)
axs[0].set_title("x")

axs[1].plot(t, x)
axs[1].stem(t, x)
axs[1].set_title("x")

# plt.tight_layout()
# plt.show()

# b
t = np.linspace(0, 3, 16000)  
zoom=200          
t = np.linspace(0, 1/zoom, int(16000/zoom))
x = np.sin(800*2*np.pi*t)
fig, axs = plt.subplots(2)

axs[0].plot(t, x)
axs[0].set_title("x")

axs[1].plot(t, x)
axs[1].stem(t, x)
axs[1].set_title("x")


# plt.tight_layout()
# plt.show()

# c
f = 240  #frecventa semnal             
fs = 1000 # frecventa esantioane
T = 1 / fs # perioada dintre esantioane
t = np.linspace(0, 0.01, int(0.01 / T))  

x = 2 * (f * t - np.floor(f * t)) - 1

plt.figure(figsize=(8,4))
plt.plot(t, x)  
plt.stem(t, x)
plt.grid(True)
# plt.show()

# d
f = 300          
fs = 10000 
T = 1 / fs 
t = np.linspace(0, 0.01, int(0.01 / T))  

x = np.sign(np.sin(2 * np.pi * f * t))

plt.figure(figsize=(8,4))
plt.plot(t, x)  
plt.stem(t, x)
plt.grid(True)
# plt.show()

# e
r, c = 128, 128
I = np.random.rand(r, c)  

plt.figure(figsize=(8,4))
plt.imshow(I)  
# plt.show()

#f
r, c = 128, 128

I = np.zeros((r, c))  

for i in range(r):
    for j in range(c):
        I[i, j] = (i + j) / (r + c - 2)  

plt.figure(figsize=(8,4))
plt.imshow(I)
plt.show()
