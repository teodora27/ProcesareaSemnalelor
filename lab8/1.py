import numpy as np
import matplotlib.pyplot as plt

# a)
N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, N)

y = trend + season + noise

plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(t, trend, color='red')
plt.title('Trend')
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(t, season, color='blue')
plt.title('Sezon')
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(t, noise, color='green')
plt.title('Zgomot alb gaussian')
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(t, y, color='black')
plt.title('Seria finala')
plt.grid(True)

plt.tight_layout()
plt.show()

# b)
# autocorelatie = cat de mult seamana cu ea insasi la diferite decalaje

def autocorrelation(x):
    x = x - np.mean(x)              
    result = np.correlate(x, x, mode='full')
    result = result / result[0]       #normalizat
    return result

correlation = autocorrelation(y)
nr = np.arange(len(correlation))

plt.figure(figsize=(10,4))
plt.stem(nr[:50], correlation[:50]) 
plt.title("Vectorul de autocorelatie")
plt.xlabel("Nr")
plt.ylabel("Autocorelatie")
plt.grid(True)
plt.show()

# c)
p = 10  # ordin AR

X = []
Y = []
for t in range(p, N):
    X.append(y[t-p:t][::-1])  
    Y.append(y[t])

X = np.array(X)
Y = np.array(Y)

XT = X.T
XT_X = XT @ X
XT_Y = XT @ Y

sol = np.linalg.inv(XT_X) @ XT_Y

predictions = []
for t in range(p, N):
    y_hat = np.dot(sol, y[t-p:t][::-1])
    predictions.append(y_hat)

predictions = np.array(predictions)

plt.figure(figsize=(12,6))
plt.plot(y, label='Seria originală', color='black')
plt.plot(np.arange(p, N), predictions, label=f'Predicții AR({p})', color='orange')
plt.title(f'Model AR({p})')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()
plt.grid(True)
plt.show()

print("Coeficientii AR:", sol)

# d)
train_size = int(0.8 * N)
train, test = y[:train_size], y[train_size:]

def fit(series, p):
    X, Y = [], []
    for t in range(p, len(series)):
        X.append(series[t-p:t][::-1])
        Y.append(series[t])
    X, Y = np.array(X), np.array(Y)
    XT = X.T
    XT_X = XT @ X
    XT_Y = XT @ Y
    sol = np.linalg.inv(XT_X) @ XT_Y
    return sol

def predict(series, coef, p, start, end):
    preds = []
    for t in range(start, end+1):
        y_hat = np.dot(coef, series[t-p:t][::-1])
        preds.append(y_hat)
    return np.array(preds)

errors = []
best_p, best_mse = None, float("inf")

for p in range(1, 21):  
    coef = fit(train, p) 
    preds = predict(y, coef, p, train_size, N-1)  
    mse = np.mean((test - preds)**2)
    errors.append(mse)
    if mse < best_mse:
        best_mse, best_p = mse, p

print(f"Cel mai bun ordin AR este p={best_p} cu MSE={best_mse:.3f}")
