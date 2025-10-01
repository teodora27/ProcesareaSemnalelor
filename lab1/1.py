import numpy as np
import matplotlib.pyplot as plt

# subpunctul a
# t = np.linspace(0, 0.03, int(0.03/0.0005))

# subpunct b
t=np.linspace(0,0.03,60)
x = np.cos(520*np.pi*t + np.pi/3)
y = np.cos(280*np.pi*t - np.pi/3)
z = np.cos(120*np.pi*t + np.pi/3)

fig, axs = plt.subplots(3)
fig.suptitle("Semnale sintetice")

axs[0].plot(t, x)
axs[0].set_title("x")

axs[1].plot(t, y)
axs[1].set_title("y")

axs[2].plot(t, z)
axs[2].set_title("z")

plt.tight_layout()
plt.show()

# subpunctul c
t = np.linspace(0, 0.03, 7)            

x = np.cos(520*np.pi*t + np.pi/3)
y = np.cos(280*np.pi*t - np.pi/3)
z = np.cos(120*np.pi*t + np.pi/3)

fig, axs = plt.subplots(3)
fig.suptitle("Semnale sintetice")

axs[0].plot(t, x)
axs[0].stem(t, x)
axs[0].set_title("x")

axs[1].plot(t, y)
axs[1].stem(t, y)
axs[1].set_title("y")

axs[2].plot(t, z)
axs[2].stem(t, z)
axs[2].set_title("z")

plt.tight_layout()
plt.show()
