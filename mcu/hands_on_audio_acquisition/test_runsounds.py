import os
import soundfile as sf
import sounddevice as sd
import time

# Current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Path to the directory containing .wav files
sound_files_path = str(current_dir) + '/../../classification/src/classification/datasets/micro_sounds'

# List all files in the directory
sound_files = [f for f in os.listdir(sound_files_path) if f.endswith('.wav')]

# Play each sound file
for sound_file in sound_files:
    sound_path = os.path.join(sound_files_path, sound_file)
    print(f'Playing {sound_file}')
    # Play the sound
    data, fs = sf.read(sound_path)
    sd.play(data, fs)
    print("runnnnnn")
    time.sleep(5)

print("All sounds have been played.")