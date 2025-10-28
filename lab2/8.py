import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(-np.pi/2, np.pi/2, 1000)
sin_real = np.sin(alpha)
sin_taylor = alpha
sin_pade = (alpha - (7 * alpha**3) / 60) / (1 + alpha**2 / 20)

error_taylor = np.abs(sin_real - sin_taylor)
error_pade = np.abs(sin_real - sin_pade)

plt.figure(figsize=(10, 6))
plt.plot(alpha, sin_real, label='sin(alfa)', linewidth=2)
plt.plot(alpha, sin_taylor, '--', label='Taylor: alfa')
plt.plot(alpha, sin_pade, '--', label='Pade')
plt.xlabel('alfa [rad]')
plt.ylabel('Valoare')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(alpha, error_taylor, label='Eroare Taylor')
plt.plot(alpha, error_pade, label='Eroare Pade')
plt.title('Eroarea aproximarilor fata de sin(alfa)')
plt.xlabel('alfa [rad]')
plt.ylabel('Eroare absoluta')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(alpha, error_taylor, label='Eroare Taylor')
plt.semilogy(alpha, error_pade, label='Eroare Pade')
plt.title('Eroarea (logaritmica) a aproximarilor fata de sin(alfa)')
plt.xlabel('alfa [rad]')
plt.ylabel('Eroare absoluta (log)')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
