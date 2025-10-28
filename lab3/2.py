import numpy as np
import matplotlib.pyplot as plt
import time

fs = 5000          
f = 6              # frecventa semnalului
omega = 1          # frecventa de rotatie pe cerc
duration = 1      
N = int(fs * duration)
n = np.arange(N)
x = np.sin(2 * np.pi * f * n / fs + np.pi / 2)

z = np.exp(-2j * np.pi * omega * n / fs)
y = x * z
dist = np.abs(y)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sc = plt.scatter(n, x, c=dist, cmap='plasma', s=10)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)

plt.subplot(1, 2, 2)
sc = plt.scatter(y.real, y.imag, c=dist, cmap='plasma', s=10)
plt.xlabel("Re")
plt.ylabel("Im")
plt.title(f"Infasurare cu omega = {omega}")
plt.axis('equal')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.grid(True)
plt.colorbar(sc, label="Distanta fata de origine")

theta = np.linspace(0, 2 * np.pi, 500)
plt.plot(np.cos(theta), np.sin(theta))

plt.tight_layout()
plt.show()

omega_values = [1, 2, 3, 4]
plt.figure(figsize=(14, 10))
for i, omega in enumerate(omega_values):
    z = np.exp(-2j * np.pi * omega * n / fs)
    y = x * z
    dist = np.abs(y)

    plt.subplot(2, 2, i + 1)
    sc = plt.scatter(y.real, y.imag, c=dist, cmap='plasma', s=10)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(f"Infasurare cu omega = {omega}")
    plt.axis('equal')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid(True)
    plt.colorbar(sc, label="Distanta fata de origine")

plt.tight_layout()
plt.show()

omega = 2
z = np.exp(-2j * np.pi * omega * n / fs)
y = x * z

fig, ax = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2 * np.pi, 500)
ax.plot(np.cos(theta), np.sin(theta), 'r--', label='Cercul unitate')
ax.set_xlabel("Re")
ax.set_ylabel("Im")
ax.set_title(f"Animatie infasurare omega = {omega}")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid(True)

linie, = ax.plot([], [], color='blue', linewidth=1)
punct, = ax.plot([], [], 'ro', markersize=6)

for i in range(len(y)):
    linie.set_data(y.real[:i+1], y.imag[:i+1])
    punct.set_data([y.real[i]], [y.imag[i]]) 
    plt.pause(0.003)

plt.show()
plt.close(fig)
