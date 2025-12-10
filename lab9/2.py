import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, N)

y = trend + season + noise

def exponential_smoothing(series, alpha):
    result = np.zeros_like(series)
    result[0] = series[0]  
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result

# rezultat pentru alpha fixat
alpha = 0.2
y_smooth = exponential_smoothing(y, alpha)

plt.figure(figsize=(12,6))
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, y_smooth, label=f"alpha={alpha}", linewidth=2)
plt.legend()
plt.show()

#cautam cel mai bun alpha
def error(x, s):
    N = len(x)
    error_sum = 0.0
    
    for t in range(N-2):  
        diff = s[t] - x[t+1]
        error_sum += diff ** 2
    
    return error_sum / (N-2)

alphas = np.linspace(0.01, 0.99, 50) 
errors = []

for a in alphas:
    y_smooth = exponential_smoothing(y, a)
    mse = error(y, y_smooth)
    errors.append(mse)

best_alpha = alphas[np.argmin(errors)]
best_err = min(errors)
print(f"Alpha optim: {best_alpha:.3f}, Eroare: {best_err:.3f}")

y_best = exponential_smoothing(y, best_alpha)

plt.figure(figsize=(12,6))
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, y_best, label=f"alpha={best_alpha:.2f}", linewidth=2, color="red")
plt.legend()
plt.show()

# exponentiere dubla
def double_exponential_smoothing(series, alpha, beta):
    N = len(series)
    s = np.zeros(N)
    b = np.zeros(N)
    
    s[0] = series[0]
    b[0] = series[1] - series[0]  
    
    for t in range(1, N):
        s[t] = alpha * series[t] + (1 - alpha) * (s[t-1] + b[t-1])
        b[t] = beta * (s[t] - s[t-1]) + (1 - beta) * b[t-1]
    
    return s, b

def error_double(x, s, b):
    N = len(x)
    acc = 0.0
    for k in range(N-1):  
        x_hat_next = s[k] + b[k]
        diff = x_hat_next - x[k+1]
        acc += diff * diff
    return acc / (N-1)  

alphas = np.linspace(0.01, 0.99, 20)
betas = np.linspace(0.01, 0.99, 20)

best_err = float("inf")
best_alpha, best_beta = None, None

for a in alphas:
    for b in betas:
        s_db, b_db = double_exponential_smoothing(y, a, b)
        err = error_double(y, s_db, b_db)
        if err < best_err:
            best_err = err
            best_alpha, best_beta = a, b

print(f"Alpha optim: {best_alpha:.3f}, Beta optim: {best_beta:.3f}, Eroare: {best_err:.3f}")

s_best, b_best = double_exponential_smoothing(y, best_alpha, best_beta)
y_hat_next = s_best[:-1] + b_best[:-1]  
t_next = t[1:]                          

plt.figure(figsize=(12,6))
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, s_best, label="Nivel netezit s[t]", linewidth=2, color="red")
plt.plot(t_next, y_hat_next, label="s[t]+b[t]", linewidth=2, color="green")
plt.legend()
plt.show()

# exponentiere tripla
def triple_exponential_smoothing(series, alpha, beta, gamma, L):
    N = len(series)
    s = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    
    s[0] = series[0]
    b[0] = series[1] - series[0]
    for i in range(L):
        c[i] = series[i] - s[0]
    
    for t in range(1, N):
        if t - L >= 0:
            s[t] = alpha * (series[t] - c[t-L]) + (1 - alpha) * (s[t-1] + b[t-1])
            b[t] = beta * (s[t] - s[t-1]) + (1 - beta) * b[t-1]
            c[t] = gamma * (series[t] - s[t] - b[t-1]) + (1 - gamma) * c[t-L]
        else:
            s[t] = alpha * series[t] + (1 - alpha) * (s[t-1] + b[t-1])
            b[t] = beta * (s[t] - s[t-1]) + (1 - beta) * b[t-1]
            c[t] = series[t] - s[t]
    
    return s, b, c

def error_triple(x, s, b, c, L):
    N = len(x)
    acc = 0.0
    count = 0
    for k in range(L, N-1): 
        x_hat_next = s[k] + b[k] + c[k+1-L]
        diff = x_hat_next - x[k+1]
        acc += diff * diff
        count += 1
    return acc / count

alphas = np.linspace(0.01, 0.99, 10)
betas  = np.linspace(0.01, 0.99, 10)
gammas = np.linspace(0.01, 0.99, 10)
L = 50

best_err = float("inf")
best_alpha, best_beta, best_gamma = None, None, None

for a in alphas:
    for b in betas:
        for g in gammas:
            s, b_vec, c = triple_exponential_smoothing(y, a, b, g, L)
            err = error_triple(y, s, b_vec, c, L)
            if err < best_err:
                best_err = err
                best_alpha, best_beta, best_gamma = a, b, g

print(f"Alpha optim: {best_alpha:.3f}, Beta optim: {best_beta:.3f}, Gamma optim: {best_gamma:.3f}, Eroare: {best_err:.3f}")

s_best, b_best, c_best = triple_exponential_smoothing(y, best_alpha, best_beta, best_gamma, L)

y_hat_next = []
t_next = []
for k in range(L, N-1):
    y_hat_next.append(s_best[k] + b_best[k] + c_best[k+1-L])
    t_next.append(t[k+1])

plt.figure(figsize=(12,6))
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, s_best, label="Nivel netezit s[t]", linewidth=2, color="red")
plt.plot(t_next, y_hat_next, label="s[t]+b[t]+c[t+1-L]", linewidth=2, color="green")
plt.legend()
plt.show()
