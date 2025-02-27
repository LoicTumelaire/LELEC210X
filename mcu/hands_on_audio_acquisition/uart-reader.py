"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
import soundfile as sf
from serial.tools import list_ports
import os
import sounddevice as sd
import time

PRINT_PREFIX = "SND:HEX:"
FREQ_SAMPLING = 10200
VAL_MAX_ADC = 4096
VDD = 3.3

###################################
# File path for the sounds
###################################

current_dir = os.path.dirname(os.path.realpath(__file__))
sound_files_path = str(current_dir) + '/../../classification/src/classification/datasets/micro_sounds'
audio_file = str(current_dir) + '/audio_files'
sound_files = [f for f in os.listdir(sound_files_path) if f.endswith('.wav')]

###################################

def playsound(sound_file):
    """
        Play a sound file
    Args:
        sound_file (str): The name of the sound file to be
        played

    Returns:
        None
    """
    sound_path = os.path.join(sound_files_path, sound_file)
    print(f'Playing {sound_file}')
    # Play the sound
    data, fs = sf.read(sound_path)
    sd.play(data, fs)


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=1042).decode("ascii")
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder("<")
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array


def generate_audio(buf, file_name):
    buf = np.asarray(buf, dtype=np.float64)
    buf = buf - np.mean(buf)
    buf /= max(abs(buf))
    sf.write( audio_file + '/' + file_name + ".wav", buf, FREQ_SAMPLING)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    if args.port is None:
        print(
            "No port specified, here is a list of serial communication port available"
        )
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print("Launch this script with [-p PORT_REF] to access the communication port")

    else:
        # Play the first sound
        index = 0
        playsound(sound_files[index])

        #plt.figure(figsize=(10, 5))
        input_stream = reader(port=args.port)
        msg_counter = 0

        for msg in input_stream:
            print(f"Acquisition #{msg_counter}")
            if False:
                #Plot fft of the signal
                buffer_size = len(msg)
                times = np.linspace(0, buffer_size - 1, buffer_size) * 1 / FREQ_SAMPLING
                voltage_mV = msg * VDD / VAL_MAX_ADC * 1e3
                
                fft = np.fft.fft(voltage_mV/500)
                freqs = np.fft.fftfreq(buffer_size, 1 / FREQ_SAMPLING)
                plt.subplot(2, 1, 1)
                plt.plot(times, voltage_mV)
                plt.title(f"Acquisition #{msg_counter}")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (mV)")
                plt.ylim([0, 3300])

                plt.subplot(2, 1, 2)
                plt.plot(freqs, np.abs(fft))
                plt.title(f"FFT of acquisition #{msg_counter}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude")
                plt.xlim([50, 5000])

                
                plt.draw()
                plt.pause(0.1)
                plt.cla()

            generate_audio(msg, f"micro-{sound_files[index][:-4]}")

            msg_counter += 1
            index += 1
            if index >= len(sound_files):
                # Stop the script
                print("End of the script")
                break
            playsound(sound_files[index])
