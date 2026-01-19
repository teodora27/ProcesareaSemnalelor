import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def downsample(y, sr, target_sr=5512.5, plot_fft=False, save_output=False):
    # target_sr (Target sampeling rate) = 5512.5
    ratio = target_sr / sr
    N_new = int(len(y) * ratio)
    orig_positions = np.arange(N_new) * (sr / target_sr)

    # Linear interpolation
    y_down = np.zeros(N_new)
    for i, pos in enumerate(orig_positions):
        left = int(np.floor(pos))
        right = min(left + 1, len(y) - 1)
        alpha = pos - left
        y_down[i] = (1 - alpha) * y[left] + alpha * y[right]

    # Nyquist theorem check
    fft_orig = np.abs(np.fft.rfft(y))
    nyquist_target = target_sr / 2
    freqs_orig = np.fft.rfftfreq(len(y), d=1/sr)

    # threshold = 1% of the maximum amplitude
    # use threshold to filter noise
    threshold = np.max(fft_orig) * 0.01
    significant_freqs = freqs_orig[fft_orig > threshold]
    max_freq_orig = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

    print(f"Original SR: {sr}, Downsampled SR: {target_sr}")

    if max_freq_orig > nyquist_target:
        print(f"Nyquist frequency: {nyquist_target:.1f} Hz")
        print(f"Max significant frequency in original: {max_freq_orig:.1f} Hz")
        print("WARNING: Nyquist theorem is NOT respected! Aliasing may occur.")
    else:
        print("Nyquist theorem respected. Safe downsampling.")

    if plot_fft:
        plt.figure(figsize=(10,5))
        plt.plot(freqs_orig, fft_orig, label="Original")
        fft_down = np.abs(np.fft.rfft(y_down))
        freqs_down = np.fft.rfftfreq(len(y_down), d=1/target_sr)
        plt.plot(freqs_down, fft_down, label="Downsampled")
        plt.legend()
        plt.title("Frequency Spectrum Comparison")
        plt.show()

    if save_output:
        sf.write("downsampled.wav", y_down, int(target_sr))

    return y_down, target_sr

def hanning_window(N):
    n = np.arange(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    return w

def compute_STFT(y, sr, window, win_length=512, n_fft=4096, hop_length=32):
    N_frames = 1 + (len(y) - win_length) // hop_length
    K_bins = n_fft // 2 + 1

    STFT = np.zeros((K_bins, N_frames), dtype=complex)

    for n in range(N_frames):
        start = n * hop_length
        frame = y[start:start + win_length]

        frame = frame * window

        # Zero pading
        frame_padded = np.zeros(n_fft)
        frame_padded[:win_length] = frame

        spectrum = np.fft.rfft(frame_padded)
        STFT[:, n] = spectrum

    freqs = np.linspace(0, sr/2, K_bins)
    times = np.arange(N_frames) * hop_length / sr

    return STFT, freqs, times

def compute_STFT_spectrogram(y, sr, win_length=512, n_fft=4096, hop_length=32):
    """
    Plots the STFT spectogram.
    """
    window = hanning_window(win_length)

    # Use the core STFT function
    STFT, freqs, times = compute_STFT(y, sr, window, win_length, n_fft, hop_length)

    # Magnitude spectrogram
    M = np.abs(STFT)

    plt.figure(figsize=(10, 6))
    plt.imshow(20*np.log10(M + 1e-6), aspect='auto', origin='lower',
                extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Magnitude Spectrogram")
    plt.show()

    return M, STFT, freqs, times

def log_frequency_axis(n_semitones=780, fmin=29.1, f_ref=440):
    """
    Log-spaced freq axis: f_log(k_log) = 440 * 2^{(22+klog/10-69)/12}
    f_log (the frequency in Hz) 
        -> the vertical axis of the spectogram
        -> converts the vertical axis from an index counter into a frequency scale
        -> we use 10 bins/semitone
        -> k_log ranges from 0 to 780 => covers a pitch range of 78 semitones
        -> f_log ranges from 29.1Hz (A#0) to fs/2=2621.8Hz (E7)
    """
    k_log = np.arange(n_semitones)
    semitones_from_A4 = 22 + (k_log / 10.0) - 69 
    f_log = f_ref * (2 ** (semitones_from_A4 / 12))

    return f_log, k_log

def compute_window_derivative(win_length):
    """
    Hann window derivative
    w(t) = 0.5(1 - cos(2pi*t/L))
    w'(t) = 0.5 * sin(2pi*t/L) * (2pi*/L) = (pi/L) * sin(2pi*t/L)
    """
    n = np.arange(win_length)
    val = (np.pi / (win_length - 1)) * np.sin(2 * np.pi * n / (win_length - 1))
    return val

def compute_reassigned_spectrogram(y, sr, win_length=512, n_fft=4096, hop_length=32, 
                                   n_semitones=780, fmin=29.1, plot_spectogram=True):
    """
    First, compute IF using Lagrange & Marchand Eq 17-19 (Reassignment Method).
    Requires two STFTs: one with window w, one with window derivative w'.
    f_inst = k/n_fft + (1/2pi) * Im( S_w(k,n) / S_w'(k,n) ) 

    Then, computes the Reassigned Log-Frequency Spectrogram (MIF).
    """
    # Compute the Instantaneous Frequency using Reassignment Method
    window = hanning_window(win_length)
    window_prime = compute_window_derivative(win_length)
    
    STFT, _, _ = compute_STFT(y, sr, window, win_length, n_fft, hop_length)
    STFT_prime, _, _ = compute_STFT(y, sr, window_prime, win_length, n_fft, hop_length)
    
    M = np.abs(STFT)

    # Frequency of each bin center (cycles/sample)
    freqs_norm = np.fft.rfftfreq(n_fft, d=1.0) # [0, 0.5]
    freqs_grid = freqs_norm[:, np.newaxis]     # Broadcast to (K, N)
    
    mag_sq = np.real(STFT * np.conj(STFT))
    valid_mask = mag_sq > 1e-10 * np.max(mag_sq)
    
    ratio = np.zeros_like(STFT)
    ratio[valid_mask] = STFT_prime[valid_mask] / STFT[valid_mask]
    
    correction = -np.imag(ratio) / (2 * np.pi)
    IF_norm = freqs_grid + correction
    
    # Convert to Hz
    IF_Hz = IF_norm * sr
    IF_Hz = np.clip(IF_Hz, 0, sr/2)

    # Reassign to Log-Frequency Grid (M -> MIF)
    f_log, _ = log_frequency_axis(n_semitones, fmin)
    K_lin, N_frames = M.shape
    K_log = len(f_log)
    MIF = np.zeros((K_log, N_frames))
    
    for n in range(N_frames):
        for k_lin in range(K_lin):
            f_if = IF_Hz[k_lin, n]
            dist = np.abs(f_log - f_if)
            k_log_nearest = np.argmin(dist)
            MIF[k_log_nearest, n] += M[k_lin, n]

    if plot_spectogram:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        im1 = ax1.imshow(20*np.log10(M+1e-6), aspect='auto', origin='lower')
        ax1.set_title('Original Magnitude Spectrogram')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(20*np.log10(MIF+1e-6), aspect='auto', origin='lower')
        ax2.set_title('Reassigned Magnitude Spectrogram MIF (log freq)')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()
            
    return MIF, IF_Hz, M, f_log

def find_peaks(signal, height=None, distance=None):
    """
    Find peaks in a 1D signal.
    signal: 1D array of values
    height: minimum value to count as a peak
    distance: minimum number of samples between peaks
    """
    peaks = []
    last_peak_idx = -distance if distance else -999999
    
    for i in range(1, len(signal) - 1):
        current_val = signal[i]
        if signal[i-1] < current_val and current_val >= signal[i+1]:
            
            if height is not None and current_val < height:
                continue
                
            if distance is not None:
                if (i - last_peak_idx) < distance:
                    if signal[i] > signal[last_peak_idx]:
                        peaks.pop()
                        peaks.append(i)
                        last_peak_idx = i
                    continue
            
            peaks.append(i)
            last_peak_idx = i
            
    return np.array(peaks)

def compute_onset_detection_function(MIF, hop_length_seconds=None, plot_onset=True):
    """
    Implements the onset detection pipeline:
    1. Create kernel M_K 
    2. Convolve MIF with M_K 
    3. Compute frame-wise maximum alpha_On (onset function)
    4. Peak picking with threshold
    """
    vec_freq = np.array([[0.3], [1.0], [0.3]]) 
    vec_time = np.array([[1, 1, 1, 0, -1, -1, -1]]) 
    M_K = vec_freq @ vec_time  
    
    MIF_K = convolve2d(MIF, M_K, mode='same', boundary='fill', fillvalue=0)
    
    alpha_On = np.max(MIF_K, axis=0) 
    
    global_max = np.max(alpha_On)
    threshold = 0.2 * global_max
    
    onset_indices = find_peaks(alpha_On, height=threshold, distance=5)
    
    onset_times = None
    if hop_length_seconds is not None:
        onset_times = onset_indices * hop_length_seconds
    
    if plot_onset:
        times_axis = np.arange(len(alpha_On)) * hop_length_seconds if hop_length_seconds else np.arange(len(alpha_On))
        plt.figure(figsize=(12, 5))
        plt.plot(times_axis, alpha_On, 'b-', label='Onset Function alpha_On')
        if len(onset_indices) > 0:
            plt.plot(times_axis[onset_indices], alpha_On[onset_indices], 'ro', label='Detected Onsets')
        plt.hlines(threshold, times_axis[0], times_axis[-1], colors='g', linestyles='--', label='Threshold (20% max)')
        
        plt.title('Onset Detection Function')
        plt.xlabel('Time (s)' if hop_length_seconds else 'Frame Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    return alpha_On, onset_indices, onset_times, MIF_K

# Main:
filename = "example2.wav"
y, sr = sf.read(filename)
y_down, sr_down = downsample(y, sr)

M, STFT, freqs, times = compute_STFT_spectrogram(y_down, sr_down)

hop_len = 32
MIF, IF_Hz, M, f_log = compute_reassigned_spectrogram(
    y_down, sr_down, win_length=512, n_fft=4096, hop_length=hop_len, plot_spectogram=True)

hop_sec = hop_len / sr_down
alpha_On, onset_indices, onset_times, MIF_K = compute_onset_detection_function(MIF, hop_length_seconds=hop_sec, plot_onset=True)
