import numpy as np
import matplotlib.pyplot as plt

fs = 1000         
f = 50            
duration = 1       
t = np.linspace(0, duration, int(fs * duration))
x = np.sin(2 * np.pi * f * t)

x_decimated_a = x[::4]        
t_decimated_a = t[::4]

x_decimated_b = x[1::4]      
t_decimated_b = t[1::4]

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original')
plt.title('Semnal original (1000 Hz)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_decimated_a, x_decimated_a, 'o-', label='Decimat (start index 0)')
plt.title('Semnal decimat la 1/4 (pornind de la primul element)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_decimated_b, x_decimated_b, 'o-', label='Decimat (start index 1)')
plt.title('Semnal decimat la 1/4 (pornind de la al doilea element)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
