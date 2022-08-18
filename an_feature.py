import numpy as np
import scipy
from scipy.io import wavfile

def hann(FFT_size=1024):
    w = np.ndarray(FFT_size)
    for i in range(FFT_size):
        alpha = (2*np.pi*i/(FFT_size - 1))
        w[i] = (1 - np.cos(alpha))/2
    return w

def fft_v(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")
        
    N_min = min(N, 2)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])
    return X.ravel()


def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)
    
def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def frame_audio(audio, FFT_size=1024, hop_len=400, sample_rate=44100):
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_num = int((len(audio) - FFT_size) / hop_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*hop_len:n*hop_len+FFT_size]
    
    return frames

def time_to_freq(audio_win, FFT_size=1024):

    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F') # Empty fft matrix for compex numbers

    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft_v(audio_winT[:, n])[:audio_fft.shape[0]] # Filling the FFT matrix for the first audio_fft.shape[0] elements

    return np.transpose(audio_fft) # Return frequency domain matrix

def get_filter_points(fmin, fmax, filter_num, FFT_size, sample_rate=44100):
    
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax) # Change frequency domain to mel domain
    
    mels = np.linspace(fmin_mel, fmax_mel, num=filter_num+2) # Dividing from 0 to a maximum of (filter_num+2) parts
    freqs = mel_to_freq(mels) # Change mel domain to frequency domain
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs # Return filter points and frequencies of that points 

def get_filters(filter_points, FFT_size, mel_freqs, filter_num):
    
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1))) # Make empty filters matrix
    
    for n in range(len(filter_points)-2): # Filling filters matrix
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n]) # First filter part: creates points from 0 to 1 (filter_points[n + 1] - filter_points[n]) times. direction up /
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1]) # First filter part: creates points from 0 to 1 (filter_points[n + 2] - filter_points[n + 1]) times. direction down \
    
    enorm = 2.0 / (mel_freqs[2:filter_num+2] - mel_freqs[:filter_num]) # Normalization of filters
    filters *= enorm[:, np.newaxis] # If we wont normalize the filters, we will see the noise increase with frequency because of the filter width.

    return filters # Filters /\/\/\...

def mfcc(y, sr=44100, n_fft=1024, hop_length=400, n_mfcc=20, n_filters = 40):
    
    norm_audio = normalize_audio(y) # Normalize the audio data
    frames = frame_audio(norm_audio, FFT_size=n_fft) # Make frames
    
    window = hann(n_fft) # Make Hann windowing function
    audio_win = frames * window # windowing frames

    audio_fft = time_to_freq(audio_win=audio_win, FFT_size=n_fft) # Change time domain to frequecy domain
    
    audio_power = np.square(np.abs(audio_fft)) # Power of audio data

    freq_min = 0
    freq_max = sr / 2 # Minimum and maximum frequencies for filter points
    filter_points, mel_freqs = get_filter_points(fmin = freq_min, fmax = freq_max, filter_num=n_filters, FFT_size=n_fft, sample_rate=sr) # Points of filters and them frequencies
    filters = get_filters(filter_points=filter_points, FFT_size=n_fft, mel_freqs=mel_freqs, filter_num=n_filters) # Get filters matix
    
    audio_filtered = np.dot(filters, np.transpose(audio_power)) # filtering audio power
    audio_log = 10.0 * np.log10(audio_filtered) # Final Audio power matrix
    
    dct_filters = dct(n_mfcc, n_filters) # Making Discrete Cosine Transform Filters
    cepstral_coefficents = np.dot(dct_filters, audio_log) # Filtering Audio power and get Cepstral Coeficents...

    return cepstral_coefficents, audio_log #return Cepstral coeficents and audio power for spectogram






    





