"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports

from keras import models

import os

import pandas as pd

dir = os.path.dirname(__file__)

model_dir = dir + "/../../classification/data/models/four.keras"
print("Loading model...")

model = models.load_model(model_dir)

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 24
N_MELVECS = 20
CLASSNAMES = ['chainsaw', 'crackling_fire', 'fireworks', 'gun']

dt = np.dtype(np.uint16).newbyteorder("<")


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        # print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode(
                "ascii"
            )
            # print(line)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    
    memory_size = 10
    
    memory = np.zeros((memory_size, len(CLASSNAMES)))
        
    file_predictions = pd.DataFrame(columns=["prediction", "mean_prediction", "weighted_prediction"])
    
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
        input_stream = reader(port=args.port)
        msg_counter = 0
        
        print (input_stream)

        for melvec in input_stream:
            
            melvec = melvec[4:-8]            
            
            msg_counter += 1

            # Charge notre modèle de prédiction (CNN)
            
            prediction = model.predict(melvec.reshape((N_MELVECS, MELVEC_LENGTH, 1)).T)
            
            memory[msg_counter % 10] = prediction
            
            ## exp moving avg
            alpha = 0.1  # Smoothing factor for exponential moving average
            weights = np.exp(-alpha * np.arange(len(memory)))
            weights /= weights.sum()
            weights = np.roll(weights, -msg_counter % 10)

            ## resize weights from 10 to (10, 5)
            
            weights = np.tile(weights, (len(CLASSNAMES), 1)).T
            
            weighted_memory = np.multiply(memory, weights)
            """
            mean_prediction = np.mean(memory, axis=0)
            
            mean_prediction = mean_prediction / np.sum(mean_prediction)
            
            weighted_prediction = np.mean(weighted_memory, axis=0)
            
            weighted_prediction = weighted_prediction / np.sum(weighted_prediction)
            """
            print(f"Prediction: {prediction}")
            """
            print(f"Mean Prediction: {mean_prediction}")
            print(f"Weighted Prediction: {weighted_prediction}")
            """
            print(f"Class: {CLASSNAMES[np.argmax(prediction)]}")
            
            #file_predictions = pd.concat([file_predictions, pd.DataFrame([{"prediction": prediction, "mean_prediction": mean_prediction, "weighted_prediction": weighted_prediction}])], ignore_index=True)
            
            plt.figure()
            plt.imshow(melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T, aspect="auto")
            plt.colorbar()
            plt.title(f"Mel spectrogram {msg_counter}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.tight_layout()
            #plt.savefig(f"Results_MelSpectr/mel_spectrogram_{msg_counter}.pdf")
            plt.show()
            plt.close()
            
            # file_predictions.to_csv("predictions.csv")