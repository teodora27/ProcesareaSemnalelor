import numpy as np
import matplotlib.pyplot as plt

n = 20

t = np.arange(n)
x = np.sin(2 * np.pi * t / n)   

d = 2
y = np.roll(x, d) 

X = np.fft.fft(x)
Y = np.fft.fft(y)

r0 = np.fft.ifft(X * np.conj(Y)).real
d0 = np.argmax(r0)

r1 = np.fft.ifft(X * Y).real
d1 = np.argmax(r1)

r2 = np.fft.ifft(Y / X).real
d2 = np.argmax(r2)


print("Deplasarea reală d =", d)

print("\nRezultat metoda 0: " + str(d0))
print(np.round(r0, 3))
# d0 reprezinta cu cat trebuie de[plasat y ca sa devina x
# din acest motiv se afiseaza 18 si nu 2

print("\nRezultat metoda 1: " + str(d1))
print(np.round(r1, 3))

print("\nRezultat metoda 2:"+ str(d2))
print(np.round(r2, 3))
# d2 reprezinta deplasarea circulara, d2=d
