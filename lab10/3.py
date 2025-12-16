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


# Metoda greedy
selected = []        
remaining = list(range(p))
errors = []
# alegem max_lags, iar restul vor fi 0
max_lags = 7

while remaining and len(selected) < max_lags:
    best_err = np.inf
    best_lag = None
    best_coef = None
    
    # testam fiecare lag
    for lag in remaining:
        candidate = selected + [lag]
        Y_sub = Y[:, candidate]
        
        coef = np.linalg.solve(Y_sub.T @ Y_sub, Y_sub.T @ target)
        pred = Y_sub @ coef
        mse = np.mean((target - pred)**2)
        
        if mse < best_err:
            best_err = mse
            best_lag = lag
            best_coef = coef
    
    selected.append(best_lag)
    remaining.remove(best_lag)
    errors.append(best_err)
    
    print(f"Pas {len(selected)}: lag {best_lag}, MSE={best_err:.4f}")

x_sparse = np.zeros(p)
Y_final = Y[:, selected]
coef_final = np.linalg.solve(Y_final.T @ Y_final, Y_final.T @ target)

for idx, lag in enumerate(selected):
    x_sparse[lag] = coef_final[idx]
print("Coeficienti AR sparse:", x_sparse)

pred_sparse = Y_final @ coef_final

plt.figure(figsize=(12,6))
plt.plot(t[p:], target, label='Seria originala')
plt.plot(t[p:], pred_sparse, label='Predictie AR sparse')
plt.legend()
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.title('Model AR sparse (greedy)')
plt.show()


# Metoda regularizare l1
# minimizam norma2(Yx-y)+lambda*norma1(x)
# termenul "norma2(Yx-y)" -> masoara eroarea
# temenul "lambda*norma1(x)" -> penalizeaza coef mari

# 4. functia soft-thresholding 
def soft_threshold(z, tau):
    # daca |z|< tau atunci devine 0
    # altfel, este micsorat cu tau
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

lambda_val = 0.1   
max_iter = 2000
# tolerance -> daca norma(x_new -x)<tol ne oprim
tol = 1e-6 

# Constanta Lipschitz(L) pentru gradientul norma2(Yx - y)^2 este 2*norma2(Y)^2
# Constanta Lipschitz ajuta pentru alegerea unui pas de invatare (eta) stabil
s_max = np.linalg.norm(Y, 2)
L = 2.0 * (s_max ** 2) 
eta = 1.0 / L

x_l1 = np.zeros(p)
for it in range(max_iter):
    #resid=Yx-y
    resid = Y @ x_l1 - target 
    #grad=derivata primul termen (norma2(Yx-y))
    grad = 2.0 * (Y.T @ resid)
    x_new = soft_threshold(x_l1 - eta * grad, eta * lambda_val)
    #x_l1 - eta * grad -> reprezinta un pas pe gradient
    if np.linalg.norm(x_new - x_l1) < tol:
        print(f"Stop la iteratia {it}")
        break
    x_l1 = x_new

print("Coeficienti AR sparse (L1):", x_l1)

pred_l1 = Y @ x_l1

plt.figure(figsize=(12,6))
plt.plot(t[p:], target, label='Seria originala', alpha=0.7)
plt.plot(t[p:], pred_l1, label='Predictie AR sparse (L1)', alpha=0.7)
plt.legend()
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.title('Model AR sparse cu regularizare l1')
plt.show()
