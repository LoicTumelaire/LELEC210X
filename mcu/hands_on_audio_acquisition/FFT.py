import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def fourier_transform(audio_path, name):
    # Lire le fichier .ogg directement avec soundfile
    data, sample_rate = sf.read(audio_path)
    
    # Si stéréo, on prend un seul canal
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Appliquer la transformée de Fourier
    N = len(data)
    freq = np.fft.fftfreq(N, d=1/sample_rate)
    fft_values = np.abs(fft(data))
    
    # Afficher le spectre
    plt.figure(figsize=(10, 5))
    plt.plot(freq[:N // 2], fft_values[:N // 2])  # On ne garde que les fréquences positives
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Transformée de Fourier de l'audio")
    plt.grid()
    plt.savefig('mcu/hands_on_audio_acquisition/FFT/' + name + '.png')
    plt.show()
    

# Exemple d'utilisation
# fourier_transform("votre_audio.ogg")
# Exemple d'utilisation

import os
file_path = "mcu/hands_on_audio_acquisition/audio_files/chainsaw/acq-0.ogg"

if os.path.exists(file_path):
    print("Le fichier existe.")
else:
    print("Le fichier est introuvable.")

fourier_transform("mcu/hands_on_audio_acquisition/audio_files/chainsaw/acq-0.ogg", "chainsaw")
fourier_transform("mcu/hands_on_audio_acquisition/audio_files/fire/acq-1bis.ogg", "fire")
fourier_transform("mcu/hands_on_audio_acquisition/audio_files/fireworks/acq-6.ogg", "fireworks")
fourier_transform("mcu/hands_on_audio_acquisition/audio_files/gunshot/acq-1.ogg", "gunshot")