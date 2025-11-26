import numpy as np
import matplotlib.pyplot as plt

N1, N2 = 64, 64
n1 = np.arange(N1)
n2 = np.arange(N2)
N1_grid, N2_grid = np.meshgrid(n1, n2, indexing='ij')

# f1
X1 = np.sin(2*np.pi*N1_grid + 3*np.pi*N2_grid)
Y1 = np.fft.fft2(X1)
freq_db1 = 20 * np.log10(np.abs(Y1) + 1e-6)
# in spectru obtinem 2 puncte simetrice corespunzatoare celor 2 frecvente dominante

# f2
X2 = np.sin(4*np.pi*N1_grid) + np.cos(6*np.pi*N2_grid)
Y2 = np.fft.fft2(X2)
freq_db2 = 20 * np.log10(np.abs(Y2) + 1e-6)
print(X2)

# f3
Y3 = np.zeros((N1, N2), dtype=complex)
Y3[0,5] = 1
Y3[0,N2-5] = 1
X3 = np.fft.ifft2(Y3).real
freq_db3 = 20 * np.log10(np.abs(Y3) + 1e-6)
 
# f4
Y4 = np.zeros((N1, N2), dtype=complex)
Y4[5,0] = 1
Y4[N1-5,0] = 1
X4 = np.fft.ifft2(Y4).real
freq_db4 = 20 * np.log10(np.abs(Y4) + 1e-6)

# f5
Y5 = np.zeros((N1, N2), dtype=complex)
Y5[5,5] = 1
Y5[N1-5,N2-5] = 1
X5 = np.fft.ifft2(Y5).real
freq_db5 = 20 * np.log10(np.abs(Y5) + 1e-6)

fig, axes = plt.subplots(5, 2, figsize=(10, 10))
# f1
axes[0,0].imshow(X1, cmap='gray')
axes[0,0].set_title('Imagine f1')
axes[0,1].imshow(np.fft.fftshift(freq_db1), cmap='inferno')
axes[0,1].set_title('Spectru logaritmic f1')

# f2
axes[1,0].imshow(X2, cmap='gray')
axes[1,0].set_title('Imagine f2')
axes[1,1].imshow(np.fft.fftshift(freq_db2), cmap='inferno')
axes[1,1].set_title('Spectru logaritmic f2')

# f3
axes[2,0].imshow(X3, cmap='gray')
axes[2,0].set_title('Imagine f3')
axes[2,1].imshow(np.fft.fftshift(freq_db3), cmap='inferno')
axes[2,1].set_title('Spectru logaritmic f3')

# f4
axes[3,0].imshow(X4, cmap='gray')
axes[3,0].set_title('Imagine f4')
axes[3,1].imshow(np.fft.fftshift(freq_db4), cmap='inferno')
axes[3,1].set_title('Spectru logaritmic f4')

# f5
axes[4,0].imshow(X5, cmap='gray')
axes[4,0].set_title('Imagine f5')
axes[4,1].imshow(np.fft.fftshift(freq_db5), cmap='inferno')
axes[4,1].set_title('Spectru logaritmic f5')

plt.tight_layout()
plt.show()
