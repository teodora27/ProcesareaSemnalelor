import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

N = 1000
t = np.arange(N)
trend = 0.0005 * t**2 + 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, N)
y = trend + season + noise

mu = np.mean(y)
eps = y - mu

def fit_arma(y, p, q):
    mu = np.mean(y)
    eps = y - mu
    T = len(y)
    max_lag = max(p, q)
    
    X = []
    Y = []
    for t in range(max_lag, T):
        row = []
        # termeni AR
        for i in range(1, p+1):
            row.append(y[t-i])
        # termeni MA
        for j in range(1, q+1):
            row.append(eps[t-j])
        X.append(row)
        Y.append(y[t])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # minimizam Y-X*beta
    try:
        beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    except np.linalg.LinAlgError:
        return None, np.inf, None
    
    Y_hat = X @ beta
    resid = Y - Y_hat
    mse = np.mean(resid**2)
    
    return beta, mse, Y_hat

best_mse = np.inf
best_order = None
best_params = None
best_pred = None
max_lag_best = None

for p in range(1, 21):
    for q in range(1, 21):
        params, mse, pred = fit_arma(y, p, q)
        if params is not None and mse < best_mse:
            best_mse = mse
            best_order = (p, q)
            best_params = params
            best_pred = pred
            max_lag_best = max(p, q)

print("(p,q):", best_order)
print("Coeficienti:", best_params)
print("MSE:", best_mse)

plt.figure(figsize=(12,6))
plt.plot(y, label="Serie originala")
plt.plot(np.arange(max_lag_best, N), best_pred, 
         label=f"ARMA{best_order} predictii", linestyle="--")
plt.legend()
plt.show()

# ARIMA
p, d, q = 5, 1, 5
model = ARIMA(y, order=(p, d, q))
model_fit = model.fit()
pred = model_fit.predict(start=0, end=N-1)

plt.figure(figsize=(12,6))
plt.plot(y, label="Serie originala")
plt.plot(pred, label=f"ARIMA({p},{d},{q}) predictii", linestyle="--")
plt.legend()
plt.show()