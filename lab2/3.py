import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav

def proceseaza_semnal(x, fs, t, nume):
    sd.play(x, fs)
    sd.wait()

    x_int16 = np.int16(x / np.max(np.abs(x)) * 32767)
    wav.write(f'{nume}.wav', fs, x_int16)
    rate, x_loaded = wav.read(f'{nume}.wav')

    sd.play(x_loaded, rate)
    sd.wait()

    fig, axs = plt.subplots(2)
    axs[0].plot(t, x)
    axs[0].set_title(f"{nume} - plot")
    axs[1].stem(t, x)
    axs[1].set_title(f"{nume} - stem")
    plt.tight_layout()
    plt.show()

# a: sinus 400 Hz
fs_a = 16000
duration_a = 1
t_a = np.linspace(0, duration_a, int(fs_a * duration_a))
x_a = np.sin(400 * 2 * np.pi * t_a)
proceseaza_semnal(x_a, fs_a, t_a, 'semnal_a')

# b: sinus 800 Hz
fs_b = 16000
duration_b = 1
t_b = np.linspace(0, duration_b, int(fs_b * duration_b))
x_b = np.sin(800 * 2 * np.pi * t_b)
proceseaza_semnal(x_b, fs_b, t_b, 'semnal_b')

# c: triunghiular 240 Hz
f_c = 240
fs_c = 1000
T_c = 1 / fs_c
t_c = np.linspace(0, 1, int(fs_c * 1))
x_c = 2 * (f_c * t_c - np.floor(f_c * t_c)) - 1
proceseaza_semnal(x_c, fs_c, t_c, 'semnal_c')

# d: patrat 300 Hz
f_d = 300
fs_d = 10000
T_d = 1 / fs_d
t_d = np.linspace(0, 1, int(fs_d * 1))
x_d = np.sign(np.sin(2 * np.pi * f_d * t_d))
proceseaza_semnal(x_d, fs_d, t_d, 'semnal_d')
