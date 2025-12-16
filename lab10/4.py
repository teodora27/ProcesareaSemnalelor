import numpy as np

def radacini_polinom(coef):

    n = len(coef) - 1  # gradul polinomului
    
    C = np.zeros((n, n))
    C[:, -1] = -np.array(coef[:-1])
    for i in range(1, n):
        C[i, i-1] = 1
    
    radacini = np.linalg.eigvals(C)
    return radacini

coef = [6, -5, 1] # 6-5x+x^2
print("Radacini:", radacini_polinom(coef))
