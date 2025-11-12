# a) frecventa de esantionare e fs=1 esantion/ora = 1/3600
# b) sunt 18288 esantioane => 2 ani si 1 luna
# c) frecventa maxima = fs/2 = 1/7200

# d)
import numpy as np
import matplotlib.pyplot as plt

x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, usecols=2)
N = len(x)
Fs = 1/3600  

X = np.fft.fft(x)
X_mag = np.abs(X[:N//2]) / N
#impartim la N pentru a normaliza amplitudinea
f = Fs * np.arange(0, N//2) / N

plt.figure(figsize=(10, 5))
plt.plot(f, X_mag)
plt.xlabel('Frecventa (Hz)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()

# e) Da,semanul prezinta o componenta continua deoarece media >0
print("Media este: " + str(np.mean(x)))
x_dc_removed = x - np.mean(x)
X = np.fft.fft(x_dc_removed)
X_mag = np.abs(X[:N//2]) / N
f = Fs * np.arange(0, N//2) / N
# print(X_mag[:10])
X_mag[0] = 0.0
# print(X_mag[:10])

plt.figure(figsize=(10, 5))
plt.plot(f, X_mag)
plt.xlabel('Frecventa (Hz)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()

# f)
idx_top4 = np.argsort(X_mag)[-4:][::-1]  # descrescator
for i, k in enumerate(idx_top4, 1):
    freq_hz = f[k]
    period_sec = 1/freq_hz
    print(f"{i}. f = {freq_hz:.8e} Hz, |X| = {X_mag[k]:.4f}, perioada ≈ {period_sec:.1f} s")

# g)
samples_per_month = 30 * 24  
start_idx = 1128  
end_idx = start_idx + samples_per_month

plt.figure(figsize=(12, 5))
plt.plot(np.arange(start_idx, end_idx), x[start_idx:end_idx], color='steelblue')
plt.xlabel('Index esantion (ore)')
plt.ylabel('Numar de masini')
plt.title('Trafic pe o luna ')
plt.grid(True)
plt.show()

# h)
# Presupunand ca stim ca esantioanele sunt realizate din ora in ora, 
# dar nu avem informatii despre ziua in care a inceput masuratoarea, 
# putem incerca sa deducem acest lucru analizand semnalul in timp si 
# cautand abateri semnificative fata de media traficului. Ne bazam pe 
# faptul ca in weekend traficul este mai redus comparativ cu zilele 
# lucratoare, ceea ce ne permite sa distingem intre diferitele zile 
# ale saptamanii. O abordare naturala este segmentarea semnalului in 
# fragmente de 24 de ore si compararea mediilor zilnice. In plus, pot 
# fi luate in considerare zilele de sarbatoare, cand traficul ar trebui
# sa fie vizibil mai scazut, ceea ce ajuta la identificarea perioadei 
# din an. Daca este necesar, pot fi folosite si factori suplimentari 
# precum sezonalitatea lunara: scaderea traficului in lunile de vara 
# (iulie–august, vacante) sau cresterea la inceputul toamnei (revenirea
# la scoala).

# Neajunsuri: pe intervale scurte nu exista suficiente repere, iar in 
# lipsa sarbatorilor nu se poate ancora calendarul cu precizie. Traficul 
# poate fi influentat de evenimente locale precum maratoane, lucrari sau 
# conditii meteo, care pot imita tiparele unor sarbatori si pot 
# conduce la potriviri false.

# Factori de care depinde acuratetea: 
# ->lungimea semnalului, deoarece mai multe saptamani sau luni cresc sansele
# de a observa repere clare; 
# ->raportul semnal–zgomot, care influenteaza claritatea ciclurilor zi/noapte
# si weekend/weekday; 
# ->ipoteza ca traficul e relativ constant si lipsit de anomalii majore
# ->contextul local si calendarul specific zonei, care ofera informatii 
# pentru interpretarea corecta a semnalului


# i)
# Am ales aceasta filtrare (esnationarea la 6 ore) pentru a elimina
# componenta de frecventa inalta, dar a pastra variatiile zi-noapte
x_dc = x - np.mean(x)
X = np.fft.fft(x_dc)
freqs = np.fft.fftfreq(N, d=1/Fs)  
# freqs = la ce frecventa fizica corespunde fiecare elem din X

fc = 1/21600 #esantionam o data la 6 ore
mask = np.abs(freqs) <= fc
X_filt = X * mask
x_filt = np.fft.ifft(X_filt).real
x_filt_dc = x_filt + np.mean(x)

plt.figure(figsize=(12, 5))
plt.plot(np.arange(N), x, color='lightgray', label='Semnal brut')
plt.plot(np.arange(N), x_filt_dc, color='steelblue', label='Semnal filtrat')
plt.xlabel('Index esantion')
plt.ylabel('Numar masini')
plt.title('Trafic filtrat fc = 1/(6h)')
plt.legend()
plt.grid(True)
plt.show()

X_mag = np.abs(X[:N//2]) / N
Xf_mag = np.abs(X_filt[:N//2]) / N
fpos = freqs[:N//2]

plt.figure(figsize=(12, 5))
plt.plot(fpos, X_mag, label='Inainte filtrare', alpha=0.7)
plt.plot(fpos, Xf_mag, label='Dupa filtrare', alpha=0.9)
plt.xlabel('Frecventa (Hz)')
plt.ylabel('|X(f)|')
plt.title('Spectrul semnalului inainte vs. dupa filtrare')
plt.legend()
plt.grid(True)
plt.show()

