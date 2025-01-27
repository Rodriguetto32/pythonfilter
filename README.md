# pythonfilter
filters, frequencies?? you'll see how much fun we're going to have
from scipy.signal import firwin, freqz, lfilter
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from typing import Tuple
import librosa

#télécoms Paris. eh bien, quels sont les problèmes dans les systèmes ? 
#cela peut être fait dans les sciences technologiques et la physique 
#moderne en Python. Je sais que vous avez des problèmes avec le sujet.
#Voici la solution
#mise à jour tout au long de la course
#pyhton important !!!!


# Définir le filtre FIR
# Defining the FIR filter
num = 41
cut = 0.1
fs = 1
B = firwin(num, cut, fs=fs)

# Analyser la réponse en fréquence du filtre FIR
# Analyzing the frequency response of the FIR filter
W, H = freqz(B, worN=512, fs=fs)
plt.plot(W, np.abs(H))
plt.title('Réponse en fréquence du filtre FIR')
plt.title('Frequency response of the FIR filter')
plt.xlabel('Fréquence')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

# Appliquer le filtre FIR aux signaux
# Applying the FIR filter to signals
n = np.arange(200)
num = 41
cut = 0.2
fs = 1
a = 1
x1 = np.cos(2 * np.pi * 0.1 * n)
x2 = np.cos(2 * np.pi * 0.3 * n)

B = firwin(numtaps, cutoff, fs=fs)
y1 = lfilter(B, a, x1)
y2 = lfilter(B, a, x2)

# Fonction pour limiter la bande passante
# Function to limit bandwidth
    a = 1
    B = firwin(num, band, fs=fs)
    y = lfilter(B, a, x)
    f, H = freqz(B, worN=512, fs=fs)
    display(Audio(y, rate=fs))
    return f, H, y

# Chargement des fichiers audio
# Loading audio files
audio_mono, fs1 = librosa.load('./signals/file_mono.wav', sr=None, mono=True)
audio_stereo, fs2 = librosa.load('./signals/file_stereo.wav', sr=None, mono=False)

# Vérification des fichiers WAV
# Checking WAV files

    audio, ratesample = librosa.load(wav_filename, sr=None, mono=True)
    try:
        if sr != ratesample:
            return False
        duration = librosa.get_duration(y=audio, sr=ratesample)
        if not (dur_min < duration < dur_max):
            return False
        xmax = np.max(audio)
        if not (amin < xmax < amax):
            return False
        return True
    except Exception:
        return False
# Déterminer le nombre de canaux
# Determining the number of channels
    if x.ndim == 2:
        num_channel = 2
    else:
        num_channel = 1
    

# Effet de fondu en ouverture
# Fade In effect

    dur_muest = int(dur_fadein * fs)
    y = np.copy(x)
    assert x.ndim == 1
    assert isinstance(dur_muest, int), 'Convertir en entier le nombre d’échantillons avec int()'
    assert isinstance(dur_muest, int), 'Convert the number of samples to integer with int()'
    ganancia = np.linspace(0, 1, dur_muest)
    y[:dur_muest] *= ganancia
    return y

# Effet de fondu en fermeture
# Fade Out effect
    duration = int(dur_sistem * fs)
    y = np.copy(x)
    ganance = np.linspace(1, 0, duration)
    
    if y.ndim == 1:
        y[-dur:] *= ganance
    elif y.ndim == 2:
        y[:, -dur:] *= ganance
    else:
        raise ValueError("Le signal doit être mono ou stéréo")
        raise ValueError("The signal must be mono or stereo")
    return y

# Saturation du signal
# Signal saturation

    y = np.copy(x)
    y[y > Amax] = Amax
    y[y < -Amax] = -Amax
    return y

# Conversion de mono à stéréo et vice versa
# Conversion from mono to stereo and vice versa

    assert len(x.shape) == 1, 'L’entrée doit être mono'
    assert len(x.shape) == 1, 'The input must be mono'
    y = np.stack([x, x], axis=0) #importante this
    return y


    assert len(x.shape) == 2, 'L’entrée doit être un tableau bidimensionnel'
    assert len(x.shape) == 2, 'The input must be a two-dimensional array'
    assert x.shape[0] == 2, 'Il doit y avoir deux canaux'
    assert x.shape[0] == 2, 'There must be two channels'
    y = np.mean(x, axis=0)
    return y

# Concatenation d’audios
# Concatenation of audios

    assert len(x1.shape) == len(x2.shape), "Les audios doivent être mono ou stéréo"
    assert len(x1.shape) == len(x2.shape), "The audios must be mono or stereo"
    y = np.concatenate((x1, x2), axis=-1)
    return y

# Insérer le silence dans l’audio
# Inserting silence into the audio

    dur_ini = int(init_sistem * fs)
    dur_mues = int(dur_sistem * fs)
    
    if x.ndim == 1:
        silence = np.zeros(dur_mues)
        y = np.concatenate((x[:dur_ini], silence, x[dur_ini:]), axis=0)
    elif x.ndim == 2:
        silence = np.zeros((2, dur_mues))
        y = np.concatenate((x[:, :dur_ini], silence, x[:, dur_ini:]), axis=1)
    return y

# Quantification et déquantification
# Quantization and de-quantization
 
    x_limit = np.clip(x, -1, 1)
    delta = 1 / 2**(nbits - 1)
    xq = np.floor(x_limit / delta + 0.5)
    return xq

# Dequantification et déquantification
# Dequantization and de-quantization
    delta = 1 / (2**(nbits - 1))
    xrec = xq * delta
    return xrec

# Transformée de Fourier
# Fourier Transform
t = np.linspace(0, dur, int(fs * dur), endpoint=False)  # Vecteur de temps
t = np.linspace(0, dur, int(fs * dur), endpoint=False)  # Time vector

X = np.fft.fft(x)
k = np.arange(len(x))
fd = np.arange(len(x)) / len(x)  # Vecteur de fréquence normalisée
fd = np.arange(len(x)) / len(x)  # Normalized frequency vector

# Convolution avec FFT
# Convolution with FFT
    Nout = 2**np.ceil(np.log2(N))
    Nout = int(Nout)
    return Nout

 convfft
    N = len(x) + len(h) - 1
    y = 2**int(np.ceil(np.log2(N)))
    ytrim = np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(h, N))
    return ytrim

# Détermination de la fréquence
# Frequency determination

    L = len(x)
    N = nextpow2(L) #some def
    X = np.fft.fft(x, N)
    X = X[:N // 2]
    k = np.argmax(np.abs(X))
    fd = k / N
    fpico = fd * fs
    return fpico

#3
    L = len(x)
    N = nextpow2(L) #some def
    X = np.fft.fft(x, N)
    X = X[:N // 2]
    kmin = round(fmin * N / fs)
    kmax = round(fmax * N / fs)
    X_rango = X[kmin:kmax+1]
    k_local = np.argmax(np.abs(X_rango))
    k = k_local + kmin
    fd = k / N
    fpico = fd * fs
    return fpico
