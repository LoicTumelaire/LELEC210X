import random
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy import ndarray
from scipy.signal import fftconvolve, lfilter

import scipy.signal as signal

# -----------------------------------------------------------------------------
"""
Synthesis of the classes in :
- AudioUtil : util functions to process an audio signal.
- Feature_vector_DS : Create a dataset class for the feature vectors.
"""
# -----------------------------------------------------------------------------


class AudioUtil:
    """
    Define a new class with util functions to process an audio signal.
    """

    def open(audio_file) -> Tuple[ndarray, int]:
        """
        Load an audio file.

        :param audio_file: The path to the audio file.
        :return: The audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = sf.read(audio_file)
        if sig.ndim > 1:
            sig = sig[:, 0]
        return (sig, sr)

    def play(audio):
        """
        Play an audio file.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = audio
        sd.play(sig, sr, blocking=False)

    def normalize(audio, target_dB=52) -> Tuple[ndarray, int]:
        """
        Normalize the energy of the signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param target_dB: The target energy in dB.
        """
        sig, sr = audio
        sign = sig / np.sqrt(np.sum(np.abs(sig) ** 2))
        C = np.sqrt(10 ** (target_dB / 10))
        sign *= C
        return (sign, sr)

    def resample(audio, newsr=11025) -> Tuple[ndarray, int]:
        """
        Resample to target sampling frequency.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param newsr: The target sampling frequency.
        """
        sig, sr = audio

        ### TO COMPLETE
        
        resig = signal.resample(sig, int(len(sig) * newsr / sr))
        
        return (resig, newsr)

    def pad_trunc(audio, max_ms) -> Tuple[ndarray, int]:
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param max_ms: The target length in milliseconds.
        """
        sig, sr = audio
        sig_len = len(sig)
        max_len = int(sr * max_ms / 1000)

        if sig_len > max_len:
            # Truncate the signal to the given length at random position
            # begin_len = random.randint(0, max_len)
            begin_len = 0
            sig = sig[begin_len : begin_len + max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)

            # sig = np.append([pad_begin, sig, pad_end])
            sig = np.concatenate((pad_begin, sig, pad_end))

        return (sig, sr)

    def time_shift(audio, shift_limit=0.4) -> Tuple[ndarray, int]:
        """
        Shifts the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of the transformed signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param shift_limit: The percentage (between 0.0 and 1.0) by which to circularly shift the signal.
        """
        sig, sr = audio
        sig_len = len(sig)
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (np.roll(sig, shift_amt), sr)

    def scaling(audio, scaling_limit=5) -> Tuple[ndarray, int]:
        """
        Augment the audio signal by scaling it by a random factor.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param scaling_limit: The maximum scaling factor.
        """
        sig, sr = audio

        ### TO COMPLETE
        scaling_factor = np.random.uniform(1/scaling_limit, scaling_limit)
        
        resig = sig * scaling_factor

        return resig, sr

    def add_noise(audio, sigma=0.05) -> Tuple[ndarray, int]:
        """
        Augment the audio signal by adding gaussian noise.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param sigma: Standard deviation of the gaussian noise.
        """
        sig, sr = audio

        ### TO COMPLETE
        noise = np.random.normal(0, sigma, len(sig))
        resig = sig + noise

        audio = (resig, sr)

        return audio

    def echo(audio, nechos=2) -> Tuple[ndarray, int]:
        """
        Add echo to the audio signal by convolving it with an impulse response. The taps are regularly spaced in time and each is twice smaller than the previous one.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param nechos: The number of echoes.
        """
        sig, sr = audio
        sig_len = len(sig)
        echo_sig = np.zeros(sig_len)
        echo_sig[0] = 1
        echo_sig[(np.arange(nechos) / nechos * sig_len).astype(int)] = (
            1 / 2
        ) ** np.arange(nechos)

        sig = fftconvolve(sig, echo_sig, mode="full")[:sig_len]
        return (sig, sr)

    def filter(audio, filt) -> Tuple[ndarray, int]:
        """
        Filter the audio signal with a provided filter. Note the filter is given for positive frequencies only and is thus symmetrized in the function.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param filt: The filter to apply.
        """
        sig, sr = audio

        ### TO COMPLETE
        # Filter the signal
        resig = lfilter(filt, 1, sig)

        return (resig, sr)

    def add_bg(
        audio, dataset, num_sources=1, max_ms=5000, amplitude_limit=0.1
    ) -> Tuple[ndarray, int]:
        """
        Adds up sounds uniformly chosen at random to audio.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param dataset: The dataset to sample from.
        :param num_sources: The number of sounds to add.
        :param max_ms: The maximum duration of the sounds to add.
        :param amplitude_limit: The maximum amplitude of the added sounds.
        """
        sig, sr = audio
        
        resig = sig

        ### TO COMPLETE
                
        # Choose the sounds to add
        for _ in range(num_sources):
            # Choose a sound from the dataset
            class_name = random.choice(dataset.list_classes())
            myint = random.randint(0, len(dataset.get_class_files(class_name)) - 1)
            sound = dataset.__getitem__((class_name, myint))
            sound = AudioUtil.open(sound)
            
            # Resample the sound
            sound = AudioUtil.resample(sound, sr)
            
            # Pad or truncate the sound
            sound = AudioUtil.pad_trunc(sound, max_ms)
            
            # Normalize the sound
            sound = AudioUtil.normalize(sound, target_dB=10)
            
            # Add the sound to the signal
            resig += sound[0] * np.random.uniform(0, amplitude_limit)
        
        return (resig, sr)
    
    def distorsion(audio, alpha=0.2) -> Tuple[ndarray, int]:
        """
        Add distorsion to the audio signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param alpha: The distorsion factor.
        """
        sig, sr = audio
        
        # Compute the distorsion
        resig = sig + random.uniform(-alpha, alpha) * sig**3
        
        return (resig, sr)
    
    def pitch_shift(audio, sr2=11025, n_steps=2) -> Tuple[ndarray, int]:
        """
        Shift the pitch of the audio signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param sr2: The sampling frequency.
        :param n_steps: The number of steps to shift the pitch.
        """
        sig, sr = audio
        
        # Compute the pitch shifted signal
        resig = librosa.effects.pitch_shift(sig, sr=sr, n_steps=random.randint(-n_steps, n_steps))
        
        return (resig, sr)
    
    def add_DC(audio, alpha=0.1) -> Tuple[ndarray, int]:
        """
        Add a DC component to the audio signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param alpha: The DC component amplitude.
        """
        sig, sr = audio
        
        # Add the DC component
        resig = sig + random.uniform(-alpha, alpha)
        
        return (resig, sr)
    
    def saturation(audio, alpha=0.1) -> Tuple[ndarray, int]:
        """
        Add saturation to the audio signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param alpha: The saturation factor.
        """
        sig, sr = audio
        
        # Compute the saturation
        resig = np.tanh(random.uniform(-alpha, alpha) * sig)
        
        return (resig, sr)

    def specgram(audio, Nft=512, fs2=11025) -> ndarray:
        """
        Compute a Spectrogram.

        :param aud: The audio signal as a tuple (signal, sample_rate).
        :param Nft: The number of points of the FFT.
        :param fs2: The sampling frequency.
        """
        
        ### TO COMPLETE
        
        sig, sr = audio
        
        # Compute the STFT
        
        stft = librosa.stft(sig, n_fft=Nft, hop_length=int(Nft/4))
        
        return stft

    def get_hz2mel(fs2=11025, Nft=512, Nmel=20) -> ndarray:
        """
        Get the hz2mel conversion matrix.

        :param fs2: The sampling frequency.
        :param Nft: The number of points of the FFT.
        :param Nmel: The number of mel bands.
        """
        mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        mels = mels / np.max(mels)

        return mels

    def melspectrogram(audio, Nmel=20, Nft=512, fs2=11025) -> ndarray:
        """
        Generate a Melspectrogram.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param Nmel: The number of mel bands.
        :param Nft: The number of points of the FFT.
        :param fs2: The sampling frequency.
        """
        ### TO COMPLETE
        
        # Compute the spectrogram
        stft = AudioUtil.specgram(audio, Nft, fs2)
        
        # Get the mel conversion matrix
        mels = AudioUtil.get_hz2mel(fs2, Nft, Nmel)
        
        # Compute the mel spectrogram
        melspec = np.dot(mels, np.abs(stft))

        return melspec

    def spectro_aug_timefreq_masking(
        spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1
    ) -> ndarray:
        """
        Augment the Spectrogram by masking out some sections of it in both the frequency dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent overfitting and to help the model generalise better. The masked sections are replaced with the mean value.


        :param spec: The spectrogram.
        :param max_mask_pct: The maximum percentage of the spectrogram to mask out.
        :param n_freq_masks: The number of frequency masks to apply.
        :param n_time_masks: The number of time masks to apply.
        """
        Nmel, n_steps = spec.shape
        mask_value = np.mean(spec)
        aug_spec = np.copy(spec)  # avoids modifying spec

        freq_mask_param = max_mask_pct * Nmel
        for _ in range(n_freq_masks):
            height = int(np.round(random.random() * freq_mask_param))
            pos_f = np.random.randint(Nmel - height)
            aug_spec[pos_f : pos_f + height, :] = mask_value

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            width = int(np.round(random.random() * time_mask_param))
            pos_t = np.random.randint(n_steps - width)
            aug_spec[:, pos_t : pos_t + width] = mask_value

        return aug_spec


class Feature_vector_DS:
    """
    Dataset of Feature vectors.
    """

    def __init__(
        self,
        dataset,
        Nft=512,
        nmel=20,
        duration=500,
        shift_pct=0.4,
        normalize=False,
        data_aug=None,
        pca=None,
    ):
        self.dataset = dataset
        self.Nft = Nft
        self.nmel = nmel
        self.duration = duration  # ms
        self.sr = 11025
        self.shift_pct = shift_pct  # percentage of total
        self.normalize = normalize
        self.data_aug = data_aug
        self.data_aug_factor = 1
        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]
        self.ncol = int(
            self.duration * self.sr / (1e3 * self.Nft)
        )  # number of columns in melspectrogram
        self.pca = pca

    def __len__(self) -> int:
        """
        Number of items in dataset.
        """
        return len(self.dataset) * self.data_aug_factor

    def get_audiosignal(self, cls_index: Tuple[str, int]) -> Tuple[ndarray, int]:
        """
        Get temporal signal of i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio_file = self.dataset[cls_index]
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.time_shift(aud, self.shift_pct)
        aud = AudioUtil.pad_trunc(aud, self.duration)
        if self.data_aug is not None:
            if "add_bg" in self.data_aug:
                aud = AudioUtil.add_bg(
                    aud,
                    self.dataset,
                    num_sources=1,
                    max_ms=self.duration,
                    amplitude_limit=0.1,
                )
            if "echo" in self.data_aug:
                aud = AudioUtil.echo(aud)
            if "noise" in self.data_aug:
                aud = AudioUtil.add_noise(aud, sigma=0.05)
            if "scaling" in self.data_aug:
                aud = AudioUtil.scaling(aud, scaling_limit=5)
            if "filter" in self.data_aug:
                filt = np.array([1, -0.97])
                aud = AudioUtil.filter(aud, filt)
            if "time_shift" in self.data_aug:
                aud = AudioUtil.time_shift(aud, shift_limit=0.8)
            if "distorsion" in self.data_aug:
                aud = AudioUtil.distorsion(aud, alpha=0.2)
            if "pitch_shift" in self.data_aug:
                aud = AudioUtil.pitch_shift(aud, n_steps=2)
            if "add_DC" in self.data_aug:
                aud = AudioUtil.add_DC(aud, alpha=0.1)
            if "saturation" in self.data_aug:
                aud = AudioUtil.saturation(aud, alpha=0.2)
                

        # aud = AudioUtil.normalize(aud, target_dB=10)
        aud = (aud[0] / np.max(np.abs(aud[0])), aud[1])
        return aud

    def __getitem__(self, cls_index: Tuple[str, int]) -> Tuple[ndarray, int]:
        """
        Get i'th item in dataset.

        :param cls_index: Class name and index.
        """
        aud = self.get_audiosignal(cls_index)
        sgram = AudioUtil.melspectrogram(aud, Nmel=self.nmel, Nft=self.Nft)
        if self.data_aug is not None:
            if "aug_sgram" in self.data_aug:
                sgram = AudioUtil.spectro_aug_timefreq_masking(
                    sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
                )

        sgram_crop = sgram[:, : self.ncol]
        fv = sgram_crop.flatten()  # feature vector
        if self.normalize:
            fv /= np.linalg.norm(fv)
        if self.pca is not None:
            fv = self.pca.transform([fv])[0]
        return fv

    def display(self, cls_index: Tuple[str, int]):
        """
        Play sound and display i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio = self.get_audiosignal(cls_index)
        AudioUtil.play(audio)
        plt.figure()
        plt.imshow(
            AudioUtil.melspectrogram(audio, Nmel=self.nmel, Nft=self.Nft),
            cmap="jet",
            origin="lower",
            aspect="auto",
        )
        plt.colorbar()
        plt.title(audio)
        plt.title("MEL Spectrogram")
        plt.xlabel("Mel vector")
        plt.ylabel("Frequency [Mel]")
        plt.tight_layout()
        plt.savefig("data/spectrogram.pdf", bbox_inches="tight")
        plt.show()

    def mod_data_aug(self, data_aug) -> None:
        """
        Modify the data augmentation options.

        :param data_aug: The new data augmentation options.
        """
        self.data_aug = data_aug
        self.data_aug_factor = 1
        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]