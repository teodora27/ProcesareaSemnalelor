import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, N)

y = trend + season + noise

q = 5
mu = np.mean(y)
eps = y - mu

M = N - q
X = np.zeros((M, q)) # erorile anterioare
Y = np.zeros(M) # seria shiftata q pozitii - media 

for i in range(q, N):
    Y[i-q] = y[i] - mu
    for j in range(q):
        X[i-q, j] = eps[i-j-1]

# Y=X*theta si aflam theta
# theta = (XT*X)^(-1)*XT*Y

XT = X.T
XT_X = np.dot(XT, X)
XT_Y = np.dot(XT, Y)
theta = np.linalg.inv(XT_X).dot(XT_Y)

print("Coeficienti theta MA(q):", theta)

# reconstruim seria estimata
y_hat = np.zeros(M)
for i in range(q, N):
    val = mu + eps[i]
    for j in range(q):
        val += theta[j] * eps[i-j-1]
    y_hat[i-q] = val

plt.figure(figsize=(12,6))
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t[q:], y_hat, label=f"MA({q}) estimat", linewidth=2, color="red")
plt.legend()
plt.show()
