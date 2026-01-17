import numpy as np
import matplotlib.pyplot as plt

# EXERCITIU 1
N = 1000
t = np.arange(N)

trend = 0.0005 * t**2 + 0.05 * t
season = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, N)

y = trend + season + noise

# EXERCITIU 2
L = 50
K = N - L + 1
X = np.zeros((L, K))
for i in range(L):
    for j in range(K):
        X[i, j] = y[i + j]  


# EXERCITIU 3
XXt = np.dot(X, X.T)
eigvals_XXt, eigvecs_XXt = np.linalg.eigh(XXt)
idx_XXt = np.argsort(eigvals_XXt)[::-1]
eigvals_XXt_sorted = eigvals_XXt[idx_XXt]      
eigvecs_XXt_sorted = eigvecs_XXt[:, idx_XXt]

XtX = np.dot(X.T, X)
eigvals_XtX, eigvecs_XtX = np.linalg.eigh(XtX)
idx_XtX = np.argsort(eigvals_XtX)[::-1]
eigvals_XtX_sorted = eigvals_XtX[idx_XtX]
eigvecs_XtX_sorted = eigvecs_XtX[:, idx_XtX]

U_svd, s, Vt_svd = np.linalg.svd(X, full_matrices=False)

# Observatii
# XXt si XtX au aceleasi valori proprii si vectori proprii
# s^2 = valorile proprii XXt
# vectorii proprii pt XXt = coloanele din U_svd
# vectorii proprii pt XtX = coloanele din V_svd

print("Verificare:", np.allclose(eigvals_XXt_sorted[:50], s**2))  
print("Vectorii proprii pt XXt = coloanele din U_svd: ", np.allclose(
    np.abs(eigvecs_XXt_sorted[:,0]), 
    np.abs(U_svd[:,0])))
print("Vectorii proprii pt XtX = coloanele din V_svd: ", np.allclose(
    np.abs(eigvecs_XtX_sorted[:,0]), 
    np.abs(Vt_svd[0,:])))

# EXERCITIU 4
# r = nr de componente pe care le pastram
r = 10

# matricile Xi = sigma_i * Ui (vector coloana) * ViT (vector linie)
Xi_matrices = []
for i in range(r):
    Xi = s[i] * np.outer(U_svd[:, i], Vt_svd[i, :])
    Xi_matrices.append(Xi)

# transformare mat hankel in serie de timp
# seria = media pe antidiagonale
def hankel_to_series(Xi_mat):
    L_i, K_i = Xi_mat.shape
    x_hat = np.zeros(N)
    for k in range(N):
        indices = []
        # i+j = k
        for i in range(max(0, k - K_i + 1), min(L_i, k + 1)):
            j = k - i
            indices.append(Xi_mat[i, j])
        x_hat[k] = np.mean(indices)
    return x_hat

x_components = []
for i in range(r):
    x_hat_i = hankel_to_series(Xi_matrices[i])
    x_components.append(x_hat_i)

x_reconstructed = np.zeros(N)
for xk in x_components:
    x_reconstructed += xk

plt.figure(figsize=(10, 6))

plt.plot(y, 'k-', alpha=0.7, label='Original')
plt.plot(x_reconstructed, 'r-', linewidth=2, label='Reconstruit (r=10)')
plt.title('Original vs Reconstruit')
plt.legend()

plt.tight_layout()
plt.show()