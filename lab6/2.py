import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.random.randn(N)   

y1 = np.convolve(x, x, mode='full')
y2 = np.convolve(y1, x, mode='full')
y3 = np.convolve(y2, x, mode='full')

fig, axes = plt.subplots(4, 1, figsize=(10, 8))
axes[0].plot(x)
axes[0].set_title("Semnal initial aleator")
axes[1].plot(y1)
axes[1].set_title("x * x")
axes[2].plot(y2)
axes[2].set_title("(x * x) * x")
axes[3].plot(y3)
axes[3].set_title("((x * x) * x) * x")

plt.tight_layout()
plt.show()

x_rect = np.zeros(N)
x_rect[40:60] = 1   

y1_rect = np.convolve(x_rect, x_rect, mode='full')
y2_rect = np.convolve(y1_rect, x_rect, mode='full')
y3_rect = np.convolve(y2_rect, x_rect, mode='full')

fig, axes = plt.subplots(4, 1, figsize=(10, 8))
axes[0].plot(x_rect)
axes[0].set_title("Semnal bloc rectangular")
axes[1].plot(y1_rect)
axes[1].set_title("x_rect * x_rect")
axes[2].plot(y2_rect)
axes[2].set_title("(x_rect * x_rect) * x_rect")
axes[3].plot(y3_rect)
axes[3].set_title("((x_rect * x_rect) * x_rect) * x_rect")

plt.tight_layout()
plt.show()
