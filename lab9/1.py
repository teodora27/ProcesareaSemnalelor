import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, N)

y = trend + season + noise

plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(t, trend, color='red')
plt.title('Trend')
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(t, season, color='blue')
plt.title('Sezon')
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(t, noise, color='green')
plt.title('Zgomot alb gaussian')
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(t, y, color='black')
plt.title('Seria finala')
plt.grid(True)

plt.tight_layout()
plt.show()