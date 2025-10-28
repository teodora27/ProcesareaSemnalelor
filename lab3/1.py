import numpy as np
import matplotlib.pyplot as plt

N = 8

F = np.zeros((N, N), dtype=complex)
for i in range(N):
    for j in range(N):
        F[i, j] = np.exp(-2j * np.pi * i * j / N)

F_H = np.conj(F.T)
product = np.dot(F_H, F)
identity_scaled = N * np.eye(N)  # N * identitate

# np.allclose verifica daca 2 matrici sunt aproximativ egale
print("Matricea e unitara:", np.allclose(product, identity_scaled))
print("Norma F_H*F - N*I:", np.linalg.norm(product - identity_scaled))

fig, axs = plt.subplots(N, 2, figsize=(10, 12))
n_vals = np.arange(N)

for k in range(N):
    axs[k, 0].plot(n_vals, F[k].real, marker='o')
    axs[k, 0].set_ylabel(f"Re(F[{k}])")
    axs[k, 0].grid(True)

    axs[k, 1].plot(n_vals, F[k].imag, marker='o')
    axs[k, 1].set_ylabel(f"Im(F[{k}])")
    axs[k, 1].grid(True)

plt.tight_layout()
plt.show()
