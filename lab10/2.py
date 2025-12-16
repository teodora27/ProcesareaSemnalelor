import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, N)

y = trend + season + noise

p = 10
m = N - p

Y = np.zeros((m, p))
for i in range(m):
    for j in range(p):
        Y[i, j] = y[i + p - j - 1]

target = y[p:]

Y_T = np.transpose(Y)          
YtY = Y_T @ Y                  
Yty = Y_T @ target             
# x_star = (YtY)^(-1)*Yty
x_star = np.linalg.solve(YtY, Yty)   

print("Coeficienti AR:", x_star)

pred = Y @ x_star

plt.figure(figsize=(12,6)) 
plt.plot(t[p:], target, label='Seria originala', alpha=0.7) 
plt.plot(t[p:], pred, label='Predictie AR', alpha=0.7) 
plt.legend() 
plt.xlabel('Timp') 
plt.ylabel('Valoare') 
plt.show()