import numpy as np

N = 5

deg_p = np.random.randint(2, N)
deg_q = np.random.randint(2, N)

p = np.random.randint(-5, 6, deg_p + 1)  
q = np.random.randint(-5, 6, deg_q + 1)
r_direct = np.convolve(p, q)

size = len(p) + len(q) - 1
P = np.fft.fft(p, size)
Q = np.fft.fft(q, size)
R = P * Q
r_fft = np.fft.ifft(R).real.round().astype(int)

print("p(x) =", p)
print("q(x) =", q)
print("Produs direct =", r_direct)
print("Produs FFT    =", r_fft)
